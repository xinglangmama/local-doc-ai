"""样式配置模块
包含应用程序的所有CSS样式定义
"""

# 用户消息模板
user_template = '''
<div class="chat-message user">
    <div class="message-content">
        <div class="message-header">
            <span class="user-icon">👤</span>
            <span class="user-name">您</span>
        </div>
        <div class="message-text">{{MSG}}</div>
    </div>
</div>
'''

# AI助手消息模板
bot_template = '''
<div class="chat-message bot">
    <div class="message-content">
        <div class="message-header">
            <span class="bot-icon">🤖</span>
            <span class="bot-name">AI助手</span>
        </div>
        <div class="message-text">{{MSG}}</div>
    </div>
</div>
'''

# 主要CSS样式
CSS_STYLES = """
<style>
/* 全局样式 */
.main {
    background: #ffffff;
    color: #000000;
    min-height: 100vh;
    padding-top: 0 !important;
}

/* 聊天消息样式 */
.chat-message {
    padding: 1.2rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    transition: all 0.2s ease;
}

.chat-message:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.chat-message.user {
    background: #f0f8ff;
    margin-left: 20%;
    color: #1565c0;
    border-left: 4px solid #2196f3;
}

.chat-message.bot {
    background: #faf5ff;
    margin-right: 20%;
    color: #7b1fa2;
    border-left: 4px solid #9c27b0;
}

.message-content {
    width: 100%;
}

.message-header {
    display: flex;
    align-items: center;
    margin-bottom: 0.5rem;
    font-weight: 600;
}

.user-icon, .bot-icon {
    font-size: 1.2rem;
    margin-right: 0.5rem;
}

/* 知识库标记样式 */
.knowledge-badge {
    display: inline-flex;
    align-items: center;
    background: rgba(76, 175, 80, 0.1);
    color: #4caf50;
    padding: 0.2rem 0.5rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-left: auto;
    border: 1px solid rgba(76, 175, 80, 0.3);
}

.badge-icon {
    font-size: 0.8rem;
    margin-right: 0.3rem;
}

.badge-text {
    font-size: 0.7rem;
}

.message-text {
    line-height: 1.6;
    color: inherit;
}

/* 聊天消息中的段落样式 */
.chat-message p {
    margin: 0.5rem 0;
    color: inherit;
}

/* 聊天消息中的标题样式 */
.chat-message h1, .chat-message h2, .chat-message h3, 
.chat-message h4, .chat-message h5, .chat-message h6 {
    color: inherit;
    margin: 1rem 0 0.5rem 0;
    font-weight: 600;
}

/* 聊天消息中的列表样式 */
.chat-message ul, .chat-message ol {
    margin: 0.5rem 0;
    padding-left: 1.5rem;
}

/* 聊天消息中的代码块样式 */
.chat-message code {
    background-color: rgba(0, 0, 0, 0.08);
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
    color: inherit;
    border: 1px solid rgba(0, 0, 0, 0.1);
}

/* 聊天消息中的预格式化代码块样式 */
.chat-message pre {
    background-color: rgba(0, 0, 0, 0.05);
    padding: 0.5rem;
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 6px;
    overflow-x: auto;
    margin: 0.5rem 0;
}

.chat-message pre code {
    background-color: transparent;
    padding: 0;
    color: inherit;
}

/* 聊天消息中的引用块样式 */
.chat-message blockquote {
    border-left: 4px solid rgba(0, 0, 0, 0.2);
    margin: 0.5rem 0;
    padding-left: 1rem;
    font-style: italic;
    color: #000000;
}

/* 聊天消息中的链接样式 */
.chat-message a {
    color: #007bff;
    text-decoration: underline;
}

.chat-message a:hover {
    color: #0056b3;
}

/* 聊天消息中的表格样式 */
.chat-message table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.5rem 0;
}

.chat-message th, .chat-message td {
    border: 1px solid #dee2e6;
    padding: 0.5rem;
    text-align: left;
}

.chat-message th {
    background-color: #f8f9fa;
    font-weight: bold;
}

/* 侧边栏样式 */
.css-1d391kg {
    background: #ffffff;
}

/* 按钮样式 */
.stButton > button {
    background: #007bff;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #0056b3;
    transform: translateY(-1px);
}

/* 文件上传器样式 */
.stFileUploader {
    border: 2px dashed #007bff;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    transition: all 0.2s ease;
}

.stFileUploader:hover {
    border-color: #0056b3;
    background-color: rgba(0, 123, 255, 0.05);
}

/* 输入框样式 */
.stTextInput > div > div > input {
    border-radius: 8px;
    border: 2px solid #e9ecef;
    padding: 0.75rem;
    font-size: 1rem;
}

.stTextInput > div > div > input:focus {
    border-color: #007bff;
    box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
}

/* 容器样式 */
.block-container {
    padding-top: 0.5rem;
    padding-bottom: 2rem;
}

/* 聊天容器滚动条样式 */
.chat-container {
    max-height: 500px;
    overflow-y: auto;
    padding: 1rem;
    border: 1px solid #e9ecef;
    border-radius: 12px;
    background: #ffffff;
}

/* 标题样式 */
h1, h2, h3 {
    color: #000000;
}

/* 隐藏Streamlit默认元素 */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""

def apply_styles():
    """应用CSS样式到Streamlit应用"""
    import streamlit as st
    st.markdown(CSS_STYLES, unsafe_allow_html=True)