""" Simple Flask App to Predict Iris Species"""

import json
import logging
import sys

from flask import Flask, render_template, session, redirect, url_for
from flask import request
from flask.ext.wtf import Form
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from wtforms import SubmitField, SelectField, DecimalField
from wtforms.validators import Required

#Load Iris Data
IRIS_DATA = load_iris()
FEATURES = IRIS_DATA.data
TARGET = IRIS_DATA.target
TARGET_NAMES = IRIS_DATA.target_names

def make_prediction(iris,n_neighbors):
	''' iris format: [(sepal_length), (sepal_width),(petal_length), (petal_width)]'''
	#Make Sure Types Are correct
	iris = [float(i) for i in iris]
	n_neighbors = int(n_neighbors)

	#Fit Model and get prediction and response
	knn = KNeighborsClassifier(n_neighbors=n_neighbors)
	knn.fit(FEATURES, TARGET)
	prediction = TARGET_NAMES[knn.predict(iris)][0].capitalize()
	response = {'sepal_length': iris[0], 'sepal_width': iris[1], 'petal_length': iris[2], 
	'petal_width': iris[3], 'prediction': prediction, 'n_neighbors': n_neighbors}

	return prediction, response


#Initialize Flask App
app = Flask(__name__)

#Initialize Form Class
class IrisForm(Form):
	"""Flask wtf Form to collect Iris data"""
	n_neighb = SelectField('Number of Neighbors:', \
	choices=[(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)], coerce=int)
	sepal_length = DecimalField('Sepal Length (cm):', places=2, validators=[Required()])
	sepal_width = DecimalField('Sepal Width (cm):', places=2, validators=[Required()])
	petal_length = DecimalField('Petal Length (cm):', places=2, validators=[Required()])
	petal_width = DecimalField('Petal Width (cm):', places=2, validators=[Required()])
	submit = SubmitField('Submit')


@app.route('/', methods=['GET', 'POST'])
def model():
	"""Flask Model defining / route"""
	form = IrisForm(csrf_enabled=False)
	if form.validate_on_submit():
		#Retrieve values from form
		session['sepal_length'] = form.sepal_length.data
		session['sepal_width'] = form.sepal_width.data
		session['petal_length'] = form.petal_length.data
		session['petal_width'] = form.petal_width.data
		session['n_neighb'] = form.n_neighb.data
		#Make Prediction
		iris_instance = [(session['sepal_length']), (session['sepal_width']), \
		(session['petal_length']), (session['petal_width'])]
		try:
			session['prediction'], response = make_prediction(iris_instance,session['n_neighb'])
		except:
			return render_template('404.html'), 400
		#Implement Post/Redirect/Get Pattern
		return redirect(url_for('model'))
	
	return render_template('model.html', form=form, \
	prediction=session.get('prediction'), n_neighb=session.get('n_neighb'), \
	sepal_length=session.get('sepal_length'), sepal_width=session.get('sepal_width'), \
	petal_length=session.get('petal_length'), petal_width=session.get('petal_width'))


@app.route('/api/v1', methods=['GET'])
def api():
	"""
	API That returns JSON with request params and prediction.
	curl 'http://127.0.0.1:5000/api/v1?sepal_length=2&sepal_width=2&petal_length=2&petal_width=2&n_neighb=2'
	"""
	sepal_length = request.args.get('sepal_length', '')
	sepal_width = request.args.get('sepal_width', '')
	petal_length = request.args.get('petal_length', '')
	petal_width = request.args.get('petal_width', '')
	n_neighb = request.args.get('n_neighb', '')

	try:
		sepal_length, sepal_width, petal_length, petal_width, n_neighb = \
		float(sepal_length), float(sepal_width), float(petal_length), float(petal_width), int(n_neighb)
	except ValueError:
		return 'Invalid Request Parameters\n', 400

	iris_instance = [(sepal_length), (sepal_width),(petal_length), (petal_width)]
	prediction, response = make_prediction(iris_instance,n_neighb)
	print json.dumps(response)
	return json.dumps(response)


@app.errorhandler(404)
def page_not_found(error):
	"""Error Handler for bad routes"""
	return render_template('404.html'), 404

app.secret_key = 'super_secret_key'

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

if __name__ == '__main__':
	app.run(debug=True)
