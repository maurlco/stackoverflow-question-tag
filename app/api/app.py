from flask import Flask, jsonify, request
from .tag_generator import TagGenerator

app = Flask(__name__)

@app.route('/', methods=['GET'])
def get_simple_ui_home():
    text_file = open("../index.html", "r")

    #read whole file to a string
    index_html_page = text_file.read()

    #close file
    text_file.close()

    return index_html_page


@app.route('/api/v1/generate-tags', methods=['POST'])
def generate_tags():
    # Get the JSON body of the request
    body = request.get_json()

    # Extract the title and text from the body
    title = body.get('title')
    text = body.get('text')
    tag_generator = TagGenerator()
    # Your code to generate tags from the title and text goes here
    tags =  tag_generator.generate_tag(title, text)

    return jsonify({'tags': tags})

if __name__ == '__main__':
    app.run(port=80, debug=True)