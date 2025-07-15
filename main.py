"""
このファイルは、Webアプリのメイン処理が記述されたファイルです。
"""

############################################################
# 1. ライブラリの読み込み
############################################################
# 「.env」ファイルから環境変数を読み込むための関数
from dotenv import load_dotenv
# ログ出力を行うためのモジュール
import logging
# streamlitアプリの表示を担当するモジュール
import streamlit as st
# （自作）画面表示以外の様々な関数が定義されているモジュール
import utils
# （自作）アプリ起動時に実行される初期化処理が記述された関数
from initialize import initialize
# （自作）画面表示系の関数が定義されているモジュール
import components as cn
# （自作）変数（定数）がまとめて定義・管理されているモジュール
import constants as ct


############################################################
# 2. 設定関連
############################################################
# ブラウザタブの表示文言を設定
st.set_page_config(
    page_title=ct.APP_NAME
)

# ログ出力を行うためのロガーの設定
logger = logging.getLogger(ct.LOGGER_NAME)


############################################################
# 3. 初期化処理
############################################################
try:
    # 初期化処理（「initialize.py」の「initialize」関数を実行）
    initialize()
except Exception as e:
    # エラーログの出力
    logger.error(f"{ct.INITIALIZE_ERROR_MESSAGE}\n{str(e)}")
    # 詳細なエラーメッセージの画面表示
    st.error(f"初期化処理に失敗しました。\n\n**エラーの詳細:**\n{str(e)}", icon=ct.ERROR_ICON)
    # 後続の処理を中断
    st.stop()

# アプリ起動時のログファイルへの出力
if not "initialized" in st.session_state:
    st.session_state.initialized = True
    logger.info(ct.APP_BOOT_MESSAGE)


############################################################
# 4. 初期表示
############################################################
# サイドバーの表示
cn.display_sidebar()

# タイトル表示
cn.display_app_title()

# AIメッセージの初期表示
cn.display_initial_ai_message()


############################################################
# 5. 会話ログの表示
############################################################
try:
    # 会話ログの表示
    cn.display_conversation_log()
except Exception as e:
    # エラーログの出力
    logger.error(f"{ct.CONVERSATION_LOG_ERROR_MESSAGE}\n{e}")
    # エラーメッセージの画面表示
    st.error(utils.build_error_message(ct.CONVERSATION_LOG_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    # 後続の処理を中断
    st.stop()


############################################################
# 6. チャット入力の受け付け
############################################################
# カスタムCSS and JavaScript for Shift+Enter functionality
st.markdown("""
<style>
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: var(--background-color);
        border-top: 1px solid var(--border-color);
        padding: 1rem;
        z-index: 1000;
    }
    
    .chat-input-row {
        display: flex;
        gap: 0.5rem;
        align-items: flex-end;
    }
    
    .chat-input-area {
        flex: 1;
    }
    
    .chat-send-button {
        height: 2.5rem;
        min-width: 80px;
    }
    
    .chat-help-text {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.25rem;
    }
</style>

<script>
function handleTextAreaKeyDown(event) {
    if (event.key === 'Enter' && event.shiftKey) {
        event.preventDefault();
        // Shift+Enterが押された場合、送信ボタンをクリック
        document.querySelector('[data-testid="send_button"]').click();
    }
}

// テキストエリアにイベントリスナーを追加
document.addEventListener('DOMContentLoaded', function() {
    const textArea = document.querySelector('[data-testid="chat_input"]');
    if (textArea) {
        textArea.addEventListener('keydown', handleTextAreaKeyDown);
    }
});
</script>
""", unsafe_allow_html=True)

# チャット入力エリアの作成
col1, col2 = st.columns([0.85, 0.15])

with col1:
    chat_message = st.text_area(
        label="メッセージを入力してください",
        placeholder=ct.CHAT_INPUT_HELPER_TEXT,
        height=100,
        max_chars=1000,
        key="chat_input",
        label_visibility="collapsed"
    )
    st.markdown('<div class="chat-help-text">Shift+Enterで送信、改行はEnterキーで入力できます</div>', unsafe_allow_html=True)

with col2:
    st.write("")  # スペースを作る
    st.write("")  # スペースを作る
    send_button = st.button(
        "送信",
        key="send_button",
        type="primary",
        use_container_width=True
    )

# 送信処理の判定
should_send = False
if send_button and chat_message and chat_message.strip():
    should_send = True
    # メッセージをクリアするため、セッションステートを更新
    st.session_state.chat_input = ""
    st.rerun()


############################################################
# 7. チャット送信時の処理
############################################################
if should_send:
    # ==========================================
    # 7-1. ユーザーメッセージの表示
    # ==========================================
    # ユーザーメッセージのログ出力
    logger.info({"message": chat_message, "application_mode": st.session_state.mode})

    # ユーザーメッセージを表示
    with st.chat_message("user"):
        st.markdown(chat_message)

    # ==========================================
    # 7-2. LLMからの回答取得
    # ==========================================
    # 「st.spinner」でグルグル回っている間、表示の不具合が発生しないよう空のエリアを表示
    res_box = st.empty()
    # LLMによる回答生成（回答生成が完了するまでグルグル回す）
    with st.spinner(ct.SPINNER_TEXT):
        try:
            # 画面読み込み時に作成したRetrieverを使い、Chainを実行
            llm_response = utils.get_llm_response(chat_message)
        except Exception as e:
            # エラーログの出力
            logger.error(f"{ct.GET_LLM_RESPONSE_ERROR_MESSAGE}\n{e}")
            # エラーメッセージの画面表示
            st.error(utils.build_error_message(ct.GET_LLM_RESPONSE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            # 後続の処理を中断
            st.stop()
    
    # ==========================================
    # 7-3. LLMからの回答表示
    # ==========================================
    with st.chat_message("assistant"):
        try:
            # ==========================================
            # モードが「社内文書検索」の場合
            # ==========================================
            if st.session_state.mode == ct.ANSWER_MODE_1:
                # 入力内容と関連性が高い社内文書のありかを表示
                content = cn.display_search_llm_response(llm_response)

            # ==========================================
            # モードが「社内問い合わせ」の場合
            # ==========================================
            elif st.session_state.mode == ct.ANSWER_MODE_2:
                # 入力に対しての回答と、参照した文書のありかを表示
                content = cn.display_contact_llm_response(llm_response)
            
            # AIメッセージのログ出力
            logger.info({"message": content, "application_mode": st.session_state.mode})
        except Exception as e:
            # エラーログの出力
            logger.error(f"{ct.DISP_ANSWER_ERROR_MESSAGE}\n{e}")
            # エラーメッセージの画面表示
            st.error(utils.build_error_message(ct.DISP_ANSWER_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            # 後続の処理を中断
            st.stop()

    # ==========================================
    # 7-4. 会話ログへの追加
    # ==========================================
    # 表示用の会話ログにユーザーメッセージを追加
    st.session_state.messages.append({"role": "user", "content": chat_message})
    # 表示用の会話ログにAIメッセージを追加
    st.session_state.messages.append({"role": "assistant", "content": content})