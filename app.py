from flask import Flask, render_template, request
from utils import filter
from spell_correction import SpellCorrection

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/search', method=['POST'])
def search(text, start, row):
	return render_template('search.html')
	
@app.route('/searchResults')
def searchResults():
	search_query = request.args.get("search")
    ranking = request.args.get("rank")
    countries = [request.args.get(f"country{i}") for i in range(1, 11) if request.args.get(f"country{i}") != None]
    if search_query:
        tweets, suggestions = filter(search_query, ranking, countries)
        countries = "&".join([f"country{i}="+request.args.get(f"country{i}") for i in range(1, 11) if request.args.get(f"country{i}") != None])

        if len(suggestions) == 0:
            suggestions = []
        if len(tweets) == 0:
            return render_template('search.html', search_query = search_query, error = "No tweets Found", tweets = None, suggestions=suggestions, countries=countries)
        return render_template('search.html', search_query = search_query, error= None, tweets = tweets, suggestions=suggestions, countries=countries)

    return render_template('search.html')

@app.route('/map')
def geospatial_search():
    return render_template("map.html")

@app.route('/spellcorrection', methods=["GET"])
def spell_correction():
	"""
    GET request for get_correction.
    Parameters
    ----------
    word: str
        The mispelled input word.
    Returns
    -------
    str
        The correct word.
    """
	spell_correction = SpellCorrection(data="assets/spell-errors.txt")
    text = request.args.get("word", default=None)
    correct_text = spell_correction.get_correction(text)
    return f"{text} => {correct_text}"

@app.route('/more', method=['POST'])
def more_like_this(text):
	return "More"

@app.route('auto')
def auto_complete():
	return "auto"

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 105)