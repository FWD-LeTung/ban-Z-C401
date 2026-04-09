import json
import re
import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

from agent.agent import agent, SYSTEM_PROMPT


# --- TOOL NAME MAPPING (hiển thị tên thân thiện trong UI) ---
TOOL_DISPLAY_NAMES = {
    "tool_rag_search_specific": "🔍 Tra cứu thông số kỹ thuật",
    "tool_filter_car_by_price": "💰 Lọc xe theo ngân sách",
    "tool_get_full_info": "📋 Lấy thông tin tổng quan xe",
    "tool_search_youtube_reviews": "🎬 Tìm video review YouTube",
    "tool_search_reddit_comments": "💬 Tìm bình luận Reddit",
    "tool_search_vinfast_showrooms": "📍 Tìm showroom VinFast",
}

# Regex patterns để extract YouTube video ID từ các dạng URL
_YT_PATTERNS = [
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})"),
]


def _extract_youtube_thumbnails(text: str) -> list[cl.Image]:
    """Tìm tất cả YouTube URLs trong text và tạo thumbnail Image elements."""
    seen_ids = set()
    images = []
    for pattern in _YT_PATTERNS:
        for match in pattern.finditer(text):
            video_id = match.group(1)
            if video_id not in seen_ids:
                seen_ids.add(video_id)
                images.append(
                    cl.Image(
                        url=f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
                        name=f"YouTube: {video_id}",
                        display="inline",
                    )
                )
    return images


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Tư vấn theo ngân sách",
            message="Tôi có 500 triệu, nên mua xe VinFast nào?",
            icon="/public/money.svg",
        ),
        cl.Starter(
            label="Thông số kỹ thuật",
            message="Pin VF 5 đi được bao nhiêu km?",
            icon="/public/battery.svg",
        ),
        cl.Starter(
            label="Tìm showroom",
            message="Showroom VinFast ở Hà Nội ở đâu?",
            icon="/public/location.svg",
        ),
        cl.Starter(
            label="Trải nghiệm thực tế",
            message="Người dùng VF 7 cảm nhận thế nào về xe?",
            icon="/public/review.svg",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    """Khởi tạo session mới với system prompt."""
    cl.user_session.set("messages", [SYSTEM_PROMPT])
    await cl.Message(
        content=(
            "## Xin chào! 👋\n\n"
            "Tôi là **trợ lý AI của VinFast** — giúp bạn tìm xe phù hợp nhanh chóng.\n\n"
            "| | Khả năng |\n"
            "|---|---|\n"
            "| 🚗 | Tư vấn chọn xe theo ngân sách |\n"
            "| 📊 | Tra cứu thông số kỹ thuật chi tiết |\n"
            "| 📍 | Tìm showroom / đại lý gần bạn |\n"
            "| 💬 | Tổng hợp đánh giá từ YouTube & Reddit |\n\n"
            "👇 **Chọn gợi ý bên dưới** hoặc gõ câu hỏi bất kỳ!"
        ),
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Xử lý tin nhắn từ user → chạy LangGraph agent → hiển thị kết quả."""
    messages = cl.user_session.get("messages")
    messages.append(HumanMessage(content=message.content))

    # Placeholder cho response cuối cùng
    final_msg = cl.Message(content="")
    await final_msg.send()

    final_content = ""

    # Stream events từ LangGraph agent
    async for event in agent.astream({"messages": messages}):
        for node_name, node_state in event.items():
            latest_message = node_state["messages"][-1]

            # Agent quyết định gọi TOOL → hiển thị Step
            if isinstance(latest_message, AIMessage) and latest_message.tool_calls:
                for tc in latest_message.tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["args"]
                    display_name = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)

                    async with cl.Step(
                        name=display_name,
                        type="tool",
                    ) as step:
                        step.input = json.dumps(tool_args, ensure_ascii=False, indent=2)

            # Tool trả về kết quả → cập nhật Step output
            elif node_name == "tools":
                tool_result = latest_message.content
                # Hiển thị kết quả tool trong Step cuối
                async with cl.Step(
                    name="📄 Kết quả tra cứu",
                    type="tool",
                ) as step:
                    step.output = tool_result[:500] + "..." if len(tool_result) > 500 else tool_result

            # Agent trả lời cuối cùng (không gọi tool nữa)
            elif isinstance(latest_message, AIMessage) and not latest_message.tool_calls:
                final_content = latest_message.content

    # Cập nhật message cuối cùng
    final_msg.content = final_content

    # Attach YouTube thumbnails nếu có link YouTube trong response
    yt_images = _extract_youtube_thumbnails(final_content)
    if yt_images:
        final_msg.elements = yt_images

    # Thêm feedback buttons
    final_msg.actions = [
        cl.Action(
            name="feedback_useful",
            label="👍 Hữu ích",
            payload={"value": "useful", "question": message.content},
        ),
        cl.Action(
            name="feedback_wrong",
            label="👎 Sai/Thiếu nguồn",
            payload={"value": "wrong", "question": message.content},
        ),
    ]
    await final_msg.update()

    # Cập nhật conversation history
    messages.append(AIMessage(content=final_content))
    cl.user_session.set("messages", messages)


@cl.action_callback("feedback_useful")
async def on_feedback_useful(action: cl.Action):
    """Log feedback tích cực."""
    print(f"[FEEDBACK] 👍 Useful | Q: {action.payload.get('question', '')}")
    await cl.Message(content="Cảm ơn bạn đã đánh giá! 🙏").send()
    await action.remove()


@cl.action_callback("feedback_wrong")
async def on_feedback_wrong(action: cl.Action):
    """Log feedback tiêu cực."""
    print(f"[FEEDBACK] 👎 Wrong | Q: {action.payload.get('question', '')}")
    await cl.Message(
        content="Cảm ơn phản hồi! Tôi sẽ cải thiện. Bạn có thể cho biết thêm điểm nào sai không?"
    ).send()
    await action.remove()
