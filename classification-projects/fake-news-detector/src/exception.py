import sys
import traceback

def error_message_detail(error, error_detail: sys):
    _, _, tb = error_detail.exc_info()
    file_name = tb.tb_frame.f_code.co_filename
    line_number = tb.tb_lineno

    return (
        f"Error occurred in script: [{file_name}] "
        f"at line number [{line_number}] "
        f"with error message [{str(error)}]"
    )


class CustomException(Exception):
    def __init__(self, error_message, error_detail=sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail
        )

    def __str__(self):
        return self.error_message