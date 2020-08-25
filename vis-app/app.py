#----------------------------------------------------------------------------#
# Imports
#----------------------------------------------------------------------------#

from flask import Flask, render_template, request
# from flask.ext.sqlalchemy import SQLAlchemy
import logging
from logging import Formatter, FileHandler
from forms import *
from flask_basicauth import BasicAuth
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, connections
import os
import util

#----------------------------------------------------------------------------#
# App Config.
#----------------------------------------------------------------------------#

app = Flask(__name__)
app.config.from_object('config')

app.config['BASIC_AUTH_USERNAME'] = 'kelley'
app.config['BASIC_AUTH_PASSWORD'] = 'kelley'
app.config['BASIC_AUTH_FORCE'] = True

basic_auth = BasicAuth(app)


# Login required decorator.
'''
def login_required(test):
    @wraps(test)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return test(*args, **kwargs)
        else:
            flash('You need to login first.')
            return redirect(url_for('login'))
    return wrap
'''
#----------------------------------------------------------------------------#
# Controllers.
#----------------------------------------------------------------------------#


@app.route('/cluster/<id>')
def home(id):
    c_list = util.query(id)
    return render_template('pages/home.html', results=c_list) ## TODO fix this (should include distance as second param)

@app.route('/query/<doc>')
def query(doc):
    c_list = util.query_doc(doc)
    c_list = (c_list[0][:100], c_list[1][:100])
    return render_template('pages/home.html', results=zip(c_list[0], c_list[1]))

@app.route('/update', methods=["POST"]) ## TODO save to txt instead
def update_es():
    file_list = request.form["filelist"]
    doc_ids = [f.split("_")[0] for f in file_list.strip("][").split(",")]
    label = request.form["label0"]
    value = request.form["value0"]
    res = []
    for doc_id in doc_ids:
        pass
    return "updated!"


@app.route('/about')
def about():
    return render_template('pages/placeholder.about.html')


@app.route('/login')
def login():
    form = LoginForm(request.form)
    return render_template('forms/login.html', form=form)


@app.route('/register')
def register():
    form = RegisterForm(request.form)
    return render_template('forms/register.html', form=form)


@app.route('/forgot')
def forgot():
    form = ForgotForm(request.form)
    return render_template('forms/forgot.html', form=form)


# Error handlers
@app.errorhandler(500)
def internal_error(error):
    #db_session.rollback()
    return render_template('errors/500.html'), 500


@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

if not app.debug:
    file_handler = FileHandler('error.log')
    file_handler.setFormatter(
        Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    )
    app.logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.info('errors')

#----------------------------------------------------------------------------#
# Launch.
#----------------------------------------------------------------------------#

# Default port:
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100)

# Or specify port manually:
'''
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
'''
