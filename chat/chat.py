"""The main Chat app."""

import reflex as rx
from chat.components import chat, navbar


def index() -> rx.Component:
    """The main app."""
    return rx.chakra.hstack(
            rx.scroll_area(
                rx.chakra.vstack(
                navbar(),
                chat.chat(),
                chat.action_bar(),
                background_color=rx.color("mauve", 1),
                color=rx.color("mauve", 12),
                min_height="100vh",
                align_items="stretch",
                spacing="0",
                ),
                style={"width": "65%", "height": "100%"},
                scrollbars="vertical",
                type="auto" 
            ),
            rx.scroll_area(
                chat.table(),
                style={"width": "35%", "height": "100%"},
                scrollbars="vertical",
                type="auto"
            ),
            width='100%'
        )


# Add state and page to the app.
app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="violet",
    ),
)
app.add_page(index)
