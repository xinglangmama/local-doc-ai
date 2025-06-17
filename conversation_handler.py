"""对话处理模块
包含对话链创建、用户输入处理、聊天历史显示等功能
"""

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import sys
import os
import html
from styles import user_template, bot_template
from qwen_llm import QwenLLM

def get_llm():
    """
    获取LLM实例，使用Streamlit session_state和单例模式避免重复初始化
    """
    if "llm_instance" not in st.session_state:
        try:
            # 清理CUDA缓存以释放显存
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            st.session_state.llm_instance = QwenLLM()
            st.success("✅ 模型加载成功")
        except Exception as e:
            st.error(f"❌ 模型加载失败: {str(e)}")
            st.info("💡 建议：请尝试重启应用或检查GPU内存使用情况")
            raise e
    return st.session_state.llm_instance


def get_conversation_chain(vectorstore):
    """
    创建基于检索的对话链，结合Qwen2.5-1.5B-Instruct和向量检索
    参数: vectorstore - 向量存储对象
    返回: 对话检索链对象
    """
    # 获取共享的Qwen模型实例，避免重复初始化
    llm = get_llm()

    # 定义严格的提示模板，确保只基于检索到的文档内容回答
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""请严格基于以下提供的文档内容回答问题。如果文档中没有相关信息，请明确说明"根据提供的文档，我无法找到相关信息"。不要添加文档中没有的信息。

        文档内容：
        {context}

        问题：{question}

        回答："""
    )

    # 创建对话检索链，结合LLM和向量检索器
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,                                    # 使用的语言模型
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # 向量检索器，限制检索3个最相关文档
        return_source_documents=True,              # 返回源文档信息
        verbose=True,                              # 启用详细输出模式
        combine_docs_chain_kwargs={"prompt": custom_prompt},  # 使用自定义提示模板
        chain_type="stuff",                       # 使用stuff链类型，将所有文档内容合并
    )
    return conversation_chain  # 返回配置好的对话链


def handle_userinput_pdf(user_question):
    """
    处理用户问题，生成AI回答，并更新聊天界面
    参数: user_question - 用户输入的问题字符串
    """
    # 从会话状态中获取当前的聊天历史记录
    chat_history = st.session_state.chat_history
    # 调用对话链生成响应，传入用户问题和聊天历史
    response = st.session_state.conversation(
        {"question": user_question, "chat_history": chat_history}
    )
    
    # 调试信息：打印完整响应
    print(f"完整响应: {response}")
    print(f"回答内容: '{response['answer']}'")
    print(f"回答长度: {len(response['answer'])}")
    
    # 从响应中获取相关的源文档列表
    sources = response["source_documents"]
    print(f"源文档数量: {len(sources)}")
    
    # 创建集合来存储唯一的源文档名称
    source_names = set()
    # 遍历源文档，安全地提取文档名称
    for doc in sources:
        # 检查文档是否有metadata属性且包含source键
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            source_names.add(doc.metadata["source"])  # 添加真实的源文档名
        else:
            source_names.add("未知来源")  # 如果无法获取源信息，使用默认值
    # 将所有源文档名称合并为字符串
    src = "\n\n".join(source_names)
    src = f"\n\n> source : {src}"  # 格式化源文档信息
    
    # 检查回答是否为空
    answer = response["answer"].strip() if response["answer"] else "模型未生成有效回答"
    
    # 将用户问题添加到聊天历史中，标记为"user"
    st.session_state.chat_history.append(("user", user_question))
    # 将AI回答和源文档信息一起添加到聊天历史中，标记为"assistant"
    st.session_state.chat_history.append(("assistant", answer + src))


def show_history():
    """
    在界面上显示所有的聊天历史记录
    按照用户和AI的对话顺序依次显示消息
    """
    # 从会话状态中获取聊天历史列表
    chat_history = st.session_state.chat_history
    # 遍历聊天历史中的每条消息
    for message in chat_history:
        # 根据消息类型显示不同的模板
        if message[0] == "user":
            # 显示用户消息，使用用户消息模板
            # 对用户输入进行HTML转义，防止HTML注入
            escaped_message = html.escape(message[1])
            st.write(
                user_template.replace("{{MSG}}", escaped_message),
                unsafe_allow_html=True,  # 允许渲染HTML样式
            )
        elif message[0] == "assistant":
            # 显示AI回答消息，使用机器人消息模板
            
            # 对AI回复进行HTML转义，防止HTML标签被直接渲染
            escaped_message = html.escape(message[1])
            # 将换行符转换为HTML换行标签，保持格式
            escaped_message = escaped_message.replace('\n', '<br>')
            
            # 替换模板中的占位符
            template_html = bot_template.replace("{{MSG}}", escaped_message)
            template_html = template_html.replace("{{KNOWLEDGE_BADGE}}", "")
            
            st.write(
                template_html, 
                unsafe_allow_html=True  # 允许渲染HTML样式
            )


def clear_chat_history():
    """
    清空当前会话的所有聊天历史记录
    """
    st.session_state.chat_history = []  # 重置聊天历史为空列表


def initialize_session_state():
    """
    初始化会话状态变量
    """
    # 检查并初始化对话链对象
    if "conversation" not in st.session_state:
        st.session_state.conversation = None  # 初始化为空，等待文档上传后创建
    # 检查并初始化聊天历史列表
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # 初始化为空列表


def display_welcome_message():
    """
    显示欢迎消息
    """
    st.markdown("""
    <div style="
        text-align: center;
        padding: 3rem 2rem;
        background: #ffffff;
        border-radius: 20px;
        margin: 2rem 0;
        border: 2px dashed #dee2e6;
    ">
        <h3 style="color: #000000; margin-bottom: 1rem;">🎉 欢迎使用智能文档问答助手！</h3>
        <p style="color: #000000; font-size: 1.1rem; margin-bottom: 1.5rem;">
            上传您的文档，我将帮助您快速找到所需信息
        </p>
        <div style="
            display: flex;
            justify-content: center;
            gap: 2rem;
            flex-wrap: wrap;
            margin-top: 2rem;
        ">
            <div style="
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                min-width: 150px;
                border: 1px solid #dee2e6;
            ">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📄</div>
                <div style="color: #000000; font-weight: 600;">支持PDF</div>
            </div>
            <div style="
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                min-width: 150px;
                border: 1px solid #dee2e6;
            ">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📝</div>
                <div style="color: #000000; font-weight: 600;">支持Word</div>
            </div>
            <div style="
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                min-width: 150px;
                border: 1px solid #dee2e6;
            ">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🤖</div>
                <div style="color: #000000; font-weight: 600;">AI智能</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def handle_user_input(user_question):
    """
    处理用户输入的统一入口
    参数: user_question - 用户输入的问题
    """
    # 检查是否已经初始化了对话链
    if st.session_state.conversation is not None:
        # 处理用户输入并生成回答
        handle_userinput_pdf(user_question)
        # 使用Streamlit的内置重新运行来刷新页面
        st.rerun()
    else:
        # 如果没有上传文件，显示警告信息
        st.warning("📋 请先上传并处理文档，然后再开始对话！")