"""
Implements error handling for MatMojo.
"""

from pathlib.path import cwd

comptime OverflowError = MatMojoError[error_type="OverflowError"]
"""Type for overflow errors in MatMojo.

Fields:

file: The file where the error occurred.  
function: The function where the error occurred.  
message: An optional message describing the error.  
previous_error: An optional previous error that caused this error.
"""

comptime IndexError = MatMojoError[error_type="IndexError"]
"""Type for index errors in MatMojo.

Fields:

file: The file where the error occurred.  
function: The function where the error occurred.  
message: An optional message describing the error.  
previous_error: An optional previous error that caused this error.
"""

comptime KeyError = MatMojoError[error_type="KeyError"]
"""Type for key errors in MatMojo.

Fields:

file: The file where the error occurred.  
function: The function where the error occurred.  
message: An optional message describing the error.  
previous_error: An optional previous error that caused this error.
"""

comptime ValueError = MatMojoError[error_type="ValueError"]
"""Type for value errors in MatMojo.

Fields:

file: The file where the error occurred.  
function: The function where the error occurred.  
message: An optional message describing the error.  
previous_error: An optional previous error that caused this error.
"""


comptime ZeroDivisionError = MatMojoError[error_type="ZeroDivisionError"]

"""Type for divided-by-zero errors in MatMojo.

Fields:

file: The file where the error occurred.  
function: The function where the error occurred.  
message: An optional message describing the error.  
previous_error: An optional previous error that caused this error.
"""

comptime ConversionError = MatMojoError[error_type="ConversionError"]

"""Type for conversion errors in MatMojo.

Fields:

file: The file where the error occurred.  
function: The function where the error occurred.  
message: An optional message describing the error.  
previous_error: An optional previous error that caused this error.
"""

comptime HEADER_OF_ERROR_MESSAGE = """
---------------------------------------------------------------------------
MatMojoError                             Traceback (most recent call last)
"""


struct MatMojoError[error_type: String = "MatMojoError"](Stringable, Writable):
    """Base type for all MatMojo errors.

    Parameters:
        error_type: The type of the error, e.g., "OverflowError", "IndexError".

    Fields:

    file: The file where the error occurred.
    function: The function where the error occurred.
    message: An optional message describing the error.
    previous_error: An optional previous error that caused this error.
    """

    var file: String
    var function: String
    var message: Optional[String]
    var previous_error: Optional[String]

    fn __init__(
        out self,
        file: String,
        function: String,
        message: Optional[String],
        previous_error: Optional[Error],
    ):
        self.file = file
        self.function = function
        self.message = message
        if previous_error is None:
            self.previous_error = None
        else:
            self.previous_error = "\n".join(
                String(previous_error.value()).split("\n")[3:]
            )

    fn __str__(self) -> String:
        if self.message is None:
            return (
                "Traceback (most recent call last):\n"
                + '  File "'
                + self.file
                + '"'
                + " in "
                + self.function
                + "\n\n"
            )

        else:
            return (
                "Traceback (most recent call last):\n"
                + '  File "'
                + self.file
                + '"'
                + " in "
                + self.function
                + "\n\n"
                + String(Self.error_type)
                + ": "
                + self.message.value()
                + "\n"
            )

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("\n")
        writer.write(("-" * 80))
        writer.write("\n")
        writer.write(Self.error_type.ascii_ljust(47, " "))
        writer.write("Traceback (most recent call last)\n")
        writer.write('File "')
        try:
            writer.write(String(cwd()))
        except e:
            pass
        finally:
            writer.write("/")
        writer.write(self.file)
        writer.write('"\n')
        writer.write("----> ")
        writer.write(self.function)
        if self.message is None:
            writer.write("\n")
        else:
            writer.write("\n\n")
            writer.write(Self.error_type)
            writer.write(": ")
            writer.write(self.message.value())
            writer.write("\n")
        if self.previous_error is not None:
            writer.write("\n")
            writer.write(self.previous_error.value())
