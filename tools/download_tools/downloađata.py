import requests

url = 'http://example.com/path/to/vimeo90k.zip'  # Replace with the actual URL
response = requests.get(url)

with open('vimeo90k.zip', 'wb') as file:
    file.write(response.content)

print("Download completed!")