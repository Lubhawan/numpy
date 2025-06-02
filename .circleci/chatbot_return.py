def create_multi_file_message(content: str, filenames: list[str]) -> HumanMessage:
    """
    Helper function to create a HumanMessage with multiple file attachments.
    
    Args:
        content: The question or instruction for TableGPT
        filenames: List of filenames to attach
    
    Returns:
        HumanMessage with multiple attachments
    """
    return HumanMessage(
        content=content,
        additional_kwargs={
            "attachments": [Attachment(filename=fname) for fname in filenames]
        }
    )

# Usage example
files = ["data1.xlsx", "data2.csv", "report.xlsx"]
question = "Please merge these datasets and create a pivot table showing monthly trends."
message = create_multi_file_message(question, files)
