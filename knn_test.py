""" Simple Flask App to Predict Iris Species"""

import logging
import json
import sys

from flask import request
from flask import Flask, render_template, session, redirect, url_for
from flask.ext.wtf import Form
from wtforms import SubmitField, SelectField, DecimalField
from wtforms.validators import Required
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

#Load Iris Data
IRIS_DATA = load_iris()
FEATURES = IRIS_DATA.data
TARGET = IRIS_DATA.target
TARGET_NAMES = IRIS_DATA.target_names

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
		#Create array from values
		flower_instance = [(session['sepal_length']), (session['sepal_width']), \
		(session['petal_length']), (session['petal_width'])]
		#Fit model with n_neigh neighbors
		knn = KNeighborsClassifier(n_neighbors=session['n_neighb'])
		knn.fit(FEATURES, TARGET)
		#Return only the Predicted iris species
		session['prediction'] = TARGET_NAMES[knn.predict(flower_instance)][0].capitalize()
		#Implement Post/Redirect/Get Pattern
		return redirect(url_for('model'))

	return render_template('model.html', form=form, \
	prediction=session.get('prediction'), n_neighb=session.get('n_neighb'), \
	sepal_length=session.get('sepal_length'), sepal_width=session.get('sepal_width'), \
	petal_length=session.get('petal_length'), petal_width=session.get('petal_width'))

@app.route('/api/v1', methods=['GET'])
def api():
	"""API That returns JSON with request params and prediction"""
	sepal_length = request.args.get('sepal_length', '')
	sepal_width = request.args.get('sepal_width', '')
	petal_length = request.args.get('petal_length', '')
	petal_width = request.args.get('petal_width', '')
	n_neighb = request.args.get('n_neighb', '')

	try:
		sepal_length, sepal_width, petal_length, petal_width, n_neighb = \
		int(sepal_length), int(sepal_width), int(petal_length), int(petal_width), int(n_neighb)
	except ValueError:
		return 'Invalid Request Parameters\n', 400

	flower_instance = [(sepal_length), (sepal_width),(petal_length), (petal_width)]
	knn = KNeighborsClassifier(n_neighbors=n_neighb)
	knn.fit(FEATURES, TARGET)
	prediction = TARGET_NAMES[knn.predict(flower_instance)][0].capitalize()
	response = {'sepal_length': sepal_length,'sepal_width': sepal_width,'petal_length': petal_length, \
	'petal_width': petal_width,'n_neighb': n_neighb,'prediction': prediction}
	# return render_template('404.html'), 404
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
