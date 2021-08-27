import numpy as np
from flask import Flask, render_template, request
from backend_files.cleaning_text import load_dict, clean_text
from backend_files.vectorizing_text import bag_of_words_category, bag_of_words_response
from backend_files.building_models import get_model

# Backend Team (Lines 9-19)

CATEGORIES = ['password', 'conference', 'security', 'network', 'hardware']
data, docs_x, docs_y = load_dict('intents.json')
ERR_NOT_FOUND = "I don't have an answer for that yet. Please be more specific, or try with another question."

cv_cat, train_x_cat, train_y_cat = bag_of_words_category(docs_x)
model_cat = get_model(train_x_cat, train_y_cat, 'category')

holder = [(), (), (), (), ()]
for category in range(len(CATEGORIES)):
    cv, train_x, train_y, tags = bag_of_words_response(docs_x, docs_y, category)
    model = get_model(train_x, train_y, CATEGORIES[category])
    holder[category] = (model, cv)


def determine(query, model, cv):
    predictions = model.predict(cv.transform([query]).toarray())
    predicted_index = np.argmax(predictions)
    certainty = predictions.max(1)[0]

    if certainty < 0.6:
        return None
    else:
        return predicted_index


app = Flask(__name__)


@app.route("/get", methods=["POST"])
def chatbot_response():
    query = request.form["msg"]
    clean_query = clean_text(query)

    # Determine if the category of the query is supported.
    i = determine(clean_query, model_cat, cv_cat)
    if i is None:
        return ERR_NOT_FOUND

    # Determine if we have the answer to that particular question.
    tag = determine(clean_query, holder[i][0], holder[i][1])
    if tag is None:  # Not found
        return ERR_NOT_FOUND

    intent = data['intents'][CATEGORIES[i]][tag]['answer']
    source = data['intents'][CATEGORIES[i]][tag]['source']

    response = f"From {source}: <hr><br>Answer: {intent}"
    return response


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/test")
def test():
    return render_template("test.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/members")
def members():
    return render_template("members.html")


if __name__ == "__main__":
    app.run(debug=True)
