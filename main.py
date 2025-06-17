"""主界面模块
包含应用的主要界面逻辑、侧边栏和用户交互处理
"""

import streamlit as st
import os
import shutil
from styles import apply_styles
from config import config
from document_processor import (
    process_uploaded_files, 
    load_vectorstore, 
    check_vectorstore_exists,
    get_vectorstore_info
)
from conversation_handler import (
    get_conversation_chain,
    initialize_session_state,
    display_welcome_message,
    show_history,
    clear_chat_history,
    handle_user_input
)


def setup_page_config():
    """
    设置Streamlit页面的基本配置
    """
    app_config = config.get_config("app")
    st.set_page_config(
        page_title=app_config.get("title", "🤖 智能文档问答助手"),
        page_icon=app_config.get("page_icon", "🤖"),
        layout=app_config.get("layout", "wide"),
        initial_sidebar_state="expanded"
    )


def display_main_title():
    """
    显示应用主标题
    """
    app_config = config.get_config("app")
    title = app_config.get("title", "🤖 智能文档问答助手")
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="
            color: #000000;
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        ">{title}</h1>
        <p style="
            color: #000000;
            font-size: 1.2rem;
            margin: 0;
            opacity: 0.8;
        ">上传文档，开启智能对话体验</p>
    </div>
    """, unsafe_allow_html=True)


def create_sidebar():
    """
    创建侧边栏，包含文档管理和控制功能
    """
    with st.sidebar:
        # 在侧边栏顶部显示现代化标题
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <h2 style="
                color: white;
                font-size: 1.8rem;
                font-weight: 700;
                margin: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            ">📁 文档管理中心</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # 显示当前数据库状态
        display_database_status()
        
        # 文档上传和处理
        handle_document_upload()
        
        # 数据库管理
        handle_database_management()
        
        # 对话管理
        handle_conversation_management()
        
        # 设备配置
        handle_device_configuration()
        
        # 使用说明
        display_usage_instructions()


def display_database_status():
    """
    显示数据库状态
    """
    st.markdown("### 📊 系统状态")
    
    if check_vectorstore_exists():
        st.success("✅ 本地向量数据库已存在")
        
        # 获取向量数据库详细信息
        db_info = get_vectorstore_info()
        if db_info:
            # 显示chunk个数
            st.info(f"📄 文档块数量: {db_info['chunk_count']} 个")
            
            # 显示文档来源列表
            if db_info['source_files']:
                st.markdown("**📚 文档来源:**")
                for i, source in enumerate(db_info['source_files'], 1):
                    st.markdown(f"&nbsp;&nbsp;{i}. {source}")
            else:
                st.info("📝 暂无文档来源信息")
        else:
            st.warning("⚠️ 无法获取数据库详细信息")
    else:
        st.warning("⚠️ 本地向量数据库不存在")
    st.markdown("---")  # 分隔线


def handle_document_upload():
    """
    处理文档上传和处理
    """
    st.markdown("### 📤 文档上传")
    
    # 创建文件上传组件，支持多文件上传
    uploaded_files = st.file_uploader(
        "上传您的PDF或Word文档，然后选择处理方式", 
        accept_multiple_files=True,  # 允许同时上传多个文件
        type=["pdf", "docx"]        # 限制文件类型为PDF和Word文档
    )
    
    # 检查是否存在现有数据库
    has_existing_db = check_vectorstore_exists()
    
    if has_existing_db:
        st.info("📚 检测到现有知识库，您可以选择追加新文档或重新创建知识库")
    
    # 创建两列布局用于放置按钮
    col1, col2 = st.columns(2)
    
    with col1:
        # 追加文档按钮
        if st.button("➕ 追加文档", use_container_width=True, 
                    help="将新文档添加到现有知识库中，保留原有内容"):
            vectorstore = process_uploaded_files(uploaded_files, append_mode=True)
            if vectorstore is not None:
                # 基于向量数据库创建对话检索链，并保存到会话状态
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("✅ 文档追加完成，知识库已更新！")
            else:
                st.error("❌ 文档处理失败，请检查上传的文件！")
    
    with col2:
        # 重新创建数据库按钮
        if st.button("🔄 重新创建知识库", use_container_width=True,
                    help="删除现有知识库，仅使用新上传的文档创建全新知识库"):
            vectorstore = process_uploaded_files(uploaded_files, append_mode=False)
            if vectorstore is not None:
                # 基于向量数据库创建对话检索链，并保存到会话状态
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("✅ 新知识库创建完成，可以开始对话了！")
            else:
                st.error("❌ 文档处理失败，请检查上传的文件！")
    
    st.markdown("---")  # 分隔线


def handle_database_management():
    """
    处理数据库管理功能
    """
    st.markdown("### 💾 数据库管理")
    
    # 添加加载已保存数据库的按钮
    if st.button("📂 加载本地数据库", use_container_width=True):
        saved_vectorstore = load_vectorstore()
        if saved_vectorstore is not None:
            st.session_state.conversation = get_conversation_chain(saved_vectorstore)
            st.success("✅ 成功加载本地向量数据库！")
        else:
            st.error("❌ 未找到本地向量数据库文件！")
    
    # 添加删除本地数据库的按钮
    if st.button("🗑️ 删除本地数据库", type="secondary", use_container_width=True):
        delete_local_database()
    
    st.markdown("---")  # 分隔线


def delete_local_database():
    """
    删除本地数据库
    """
    try:
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")
            st.success("✅ 本地数据库删除成功！")
            
            # 清空相关的会话状态
            if "conversation" in st.session_state:
                del st.session_state.conversation
            if "vectorstore_loaded" in st.session_state:
                del st.session_state.vectorstore_loaded
            
            # 强制刷新页面以更新文档来源信息显示
            st.rerun()
        else:
            st.warning("⚠️ 本地数据库不存在！")
    except Exception as e:
        st.error(f"❌ 删除数据库失败: {str(e)}")


def reset_model():
    """
    重置模型，清理内存缓存
    """
    try:
        from qwen_llm import QwenLLM
        
        # 清理单例实例
        QwenLLM.clear_instance()
        
        # 清空会话状态中的模型实例
        if "llm_instance" in st.session_state:
            del st.session_state.llm_instance
        
        # 清空对话链
        if "conversation" in st.session_state:
            del st.session_state.conversation
            
        st.success("✅ 模型已重置，GPU内存已释放！")
        st.info("💡 下次使用时将重新加载模型")
        
    except Exception as e:
        st.error(f"❌ 重置模型失败: {str(e)}")


def handle_conversation_management():
    """
    处理对话管理功能
    """
    st.markdown("### 💬 对话管理")
    
    # 如果存在聊天历史，显示清空对话按钮
    if st.session_state.chat_history:
        # 创建清空对话按钮，点击时调用clear_chat_history函数
        st.button("🧹 清空对话历史", on_click=clear_chat_history, use_container_width=True)
    else:
        st.info("💭 暂无对话历史")
    
    # 重置模型按钮
    if st.button("🔄 重置模型", use_container_width=True, 
                help="清理模型缓存，释放GPU内存，解决内存不足问题"):
        reset_model()
    
    st.markdown("---")


def handle_device_configuration():
    """
    处理设备配置功能
    """
    st.markdown("### ⚙️ 设备配置")
    
    # 显示当前设备信息
    device_info = config.get_device_info()
    
    # 设备状态显示
    if device_info["current_device"] == "cuda":
        st.success(f"🚀 当前使用: GPU ({device_info.get('cuda_device_name', 'Unknown')})")
        if "cuda_memory_total" in device_info:
            st.info(f"💾 显存: {device_info['cuda_memory_total']}")
    else:
        st.info("💻 当前使用: CPU")
    
    # 设备模式选择
    current_mode = config.device_mode
    device_options = {
        "auto": "🔄 自动检测 (推荐)",
        "cuda": "🚀 强制使用GPU",
        "cpu": "💻 强制使用CPU"
    }
    
    selected_mode = st.selectbox(
        "选择计算设备:",
        options=list(device_options.keys()),
        format_func=lambda x: device_options[x],
        index=list(device_options.keys()).index(current_mode),
        help="自动模式会优先使用GPU（如果可用），否则使用CPU"
    )
    
    # 如果设备模式发生变化
    if selected_mode != current_mode:
        if st.button("🔧 应用设备设置", use_container_width=True):
            config.set_device_mode(selected_mode)
            st.success(f"✅ 设备模式已更新为: {device_options[selected_mode]}")
            st.info("💡 重启应用或重置模型以使新设置生效")
            st.rerun()
    
    # CUDA可用性检查
    if not device_info["cuda_available"]:
        st.warning("⚠️ 未检测到CUDA支持，仅可使用CPU模式")
        st.markdown("""
        **启用GPU加速的步骤:**
        1. 安装NVIDIA驱动
        2. 安装CUDA工具包
        3. 重新安装PyTorch GPU版本
        """)
    
    st.markdown("---")


def display_usage_instructions():
    """
    显示使用说明
    """
    st.markdown("### 📖 使用说明")
    st.markdown("""
    1. **上传文档**: 支持PDF和Word格式
    2. **处理文档**: 点击处理按钮创建向量数据库
    3. **开始对话**: 在下方输入框中提问
    4. **管理数据**: 可加载、删除本地数据库
    """)


def create_chat_area():
    """
    创建主聊天区域
    """
    st.markdown("### 💬 智能对话区域")
    
    # 创建聊天显示容器
    chat_container = st.container()
    with chat_container:
        # 添加聊天容器的CSS类
        st.markdown('<div id="chat-container" class="chat-container">', unsafe_allow_html=True)
        
        # 显示完整的聊天历史记录
        if not st.session_state.chat_history:
            # 显示欢迎消息
            display_welcome_message()
        else:
            show_history()
        
        # 关闭聊天容器div
        st.markdown('</div>', unsafe_allow_html=True)


def handle_user_input_area():
    """
    处理用户输入区域
    """
    # 创建用户输入容器
    st.markdown("---")
    user_question = st.chat_input(
        "💭 请输入您的问题...",
        key="user_input"
    )
    
    # 处理用户输入
    if user_question:
        handle_user_input(user_question)


def load_existing_vectorstore():
    """
    尝试加载已保存的向量数据库
    """
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        # 检查是否已经加载过向量数据库
        if "vectorstore_loaded" not in st.session_state:
            # 尝试从本地加载已保存的向量数据库
            saved_vectorstore = load_vectorstore()
            if saved_vectorstore is not None:
                try:
                    # 如果成功加载，创建对话链
                    st.session_state.conversation = get_conversation_chain(saved_vectorstore)
                    st.session_state.vectorstore_loaded = True
                    st.info("🎉 已加载本地保存的向量数据库！")
                except Exception as e:
                    st.error(f"❌ 创建对话链失败: {str(e)}")
                    st.session_state.vectorstore_loaded = False
            else:
                st.session_state.vectorstore_loaded = False


def main():
    """
    应用程序的主入口函数
    负责初始化界面、会话状态和处理用户交互
    """
    # 设置页面配置
    setup_page_config()
    
    # 应用样式
    apply_styles()
    
    # 显示主标题
    display_main_title()
    
    # 初始化会话状态变量
    initialize_session_state()
    
    # 尝试加载已保存的向量数据库
    load_existing_vectorstore()
    
    # 创建侧边栏
    create_sidebar()
    
    # 创建主聊天区域
    create_chat_area()
    
    # 处理用户输入区域
    handle_user_input_area()


# 程序入口点：当脚本直接运行时执行主函数
if __name__ == "__main__":
    main()  # 启动应用程序