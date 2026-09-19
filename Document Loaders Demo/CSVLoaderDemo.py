from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(r"C:\BUILDING GENERATIVE AI APPLICATIONS\Document Loaders Demo\HR_Analytics.csv")

data=loader.load()

# print(data)
# print(type(data))

print(data[0])
print(type(data))