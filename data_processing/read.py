import PyPDF2
import nltk


class ReadFiles():
  def __init__(self):
    pass

  def process_text(self, text):
    """
    Processes text by converting to lowercase and tokenizing.

    Args:
        text: The text to be processed.

    Returns:
        A list of tokens (words) after processing.
    """

    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text (split into words)
    tokens = nltk.word_tokenize(text)

    # You can add additional processing steps here, like removing punctuation or stop words

    text = " ".join(tokens)

    print(text)

    return text

  def read_pdf(self,pdf_path):
    """
      Reads a PDF file and returns its text content.

      Args:
          pdf_path: The path to the PDF file.

      Returns:
          A string containing the extracted text from the PDF file.
      """

    try:
      # Open the PDF file in read binary mode
      with open(pdf_path, 'rb') as pdf_file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        text = ""
        # Iterate through all pages in the PDF
        for page_num in range(len(pdf_reader.pages)):
          # Extract text from each page
          page = pdf_reader.pages[page_num]
          text += page.extract_text()

        return self.process_text(text)

    except FileNotFoundError:
      print(f"Error: PDF file not found at {pdf_path}")
      return ""