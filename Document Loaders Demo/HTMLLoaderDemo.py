#pip install beautifulsoup4
#pip install -U lxml

from langchain_community.document_loaders import BSHTMLLoader

loader=BSHTMLLoader(r"C:\BUILDING GENERATIVE AI APPLICATIONS\Document Loaders Demo\Test1.html")

data=loader.load()

print(data[0])
