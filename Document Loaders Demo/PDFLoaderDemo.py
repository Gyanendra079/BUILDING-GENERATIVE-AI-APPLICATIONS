from langchain_community.document_loaders import PyPDFLoader

pdf_path = r"C:\BUILDING GENERATIVE AI APPLICATIONS\Document Loaders Demo\SamplePDFFile.pdf"

loader = PyPDFLoader(pdf_path)

pages = loader.load()

# Clean the extracted text
for page in pages:
    text = page.page_content

    # Replace multiple whitespace/newline characters with a single space
    text = " ".join(text.split())

    page.page_content = text

print("=" * 80)
print("PAGE 1")
print("=" * 80)

print(pages[0].page_content)

print("=" * 80)
print("PAGE 2")
print("=" * 80)
print(pages[1].page_content)


