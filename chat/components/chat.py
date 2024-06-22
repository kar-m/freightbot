import reflex as rx

from chat.components import loading_icon
from chat.state import QA, State
import asyncio
from reflex_calendar import calendar


message_style = dict(display="inline-block", padding="1em", border_radius="8px", max_width=["30em", "30em", "50em", "50em", "50em", "50em"])


def message(qa: QA) -> rx.Component:
    """A single question/answer message.

    Args:
        qa: The question/answer pair.

    Returns:
        A component displaying the question/answer pair.
    """
    return rx.box(
        rx.box(
            rx.markdown(
                qa.question,
                background_color=rx.color("mauve", 4),
                color=rx.color("mauve", 12),
                **message_style,
            ),
            text_align="right",
            margin_top="1em",
        ),
        rx.box(
            rx.markdown(
                qa.answer,
                background_color=rx.color("accent", 4),
                color=rx.color("accent", 12),
                **message_style,
            ),
            text_align="left",
            padding_top="1em",
        ),
        width="100%",
    )


def chat() -> rx.Component:
    """List all the messages in a single conversation."""
    return rx.vstack(
        rx.box(rx.foreach(State.chats[State.current_chat], message), width="100%"),
        py="8",
        flex="1",
        width="100%",
        max_width="50em",
        padding_x="4px",
        align_self="center",
        overflow="hidden",
        padding_bottom="5em",
    )


def action_bar() -> rx.Component:
    """The action bar to send a new message."""
    return rx.center(
        rx.vstack(
            rx.hstack(rx.foreach(rx.selected_files("upload1"), rx.text)),
            rx.hstack(                
                img_upload(),
                rx.chakra.form(
                    rx.chakra.form_control(
                        rx.hstack(
                            rx.radix.text_field.root(
                                rx.radix.text_field.input(
                                    placeholder="Type something...",
                                    id="question",
                                    width=["15em", "20em", "45em", "50em", "50em", "50em"],
                                ),
                                rx.radix.text_field.slot(
                                    rx.tooltip(
                                        rx.icon("info", size=18),
                                        content="Enter a question to get a response.",
                                    )
                                ),
                            ),
                            rx.button(
                                rx.cond(
                                    (State.processing | State.uploading),
                                    loading_icon(height="1em"),
                                    rx.text("Send"),
                                ),
                                type="submit",
                            ),
                            
                            align_items="center",
                        ),
                        is_disabled=(State.processing | State.uploading),
                    ),
                    on_submit=[State.process_question, 
                                rx.clear_selected_files("upload1")],
                    reset_on_submit=True,
                ),

            )
        ),
        position="sticky",
        bottom="0",
        left="0",
        backdrop_filter="auto",
        backdrop_blur="lg",
        border_top=f"1px solid {rx.color('mauve', 3)}",
        background_color=rx.color("mauve", 2),
        align_items="stretch",
        width="100%",
    )


def img_upload() -> rx.Component:        
    return rx.hstack(
        rx.upload(
            rx.button(rx.image(src="/2192072-200.png", width='20px')),
            id="upload1",
            padding="0px",
            on_drop=State.handle_upload(rx.upload_files(upload_id="upload1"))
        ),
        #rx.button(
        #    "Upload",
        #    on_click=State.handle_upload(rx.upload_files(upload_id="upload1")),
        #)
    )


def table() -> rx.Component:
    return rx.vstack(
        calendar(),
        rx.data_editor(
            columns=State.cols,
            data=State.data,
            on_cell_edited=State.get_edited_data,
            align="end"
        ),
        rx.button(
            "Get Quotes",
            on_click=State.get_quotes()
        )
    )