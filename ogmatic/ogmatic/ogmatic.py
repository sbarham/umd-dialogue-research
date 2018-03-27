#############
#  IMPORTS  #
#############

import os
import sqlite3
from flask import Flask, g


###########################
#  WEBAPP INITIALIZATION  #
###########################

# create the web app and load the configuration from this file
app = Flask(__name__)
app.config.from_object(__name__)

# override some of the default configuration values
app.config.update(dict(
    DATABASE=os.path.join(app.root_path, 'diares.db'),
    SECRET_KEY='devkey',
    USERNAME='admin',
    PASSWORD='password'
))

# further override configuration values from a config file
# pointed to by environment
app.config.from_envvar('OGMATIC_SETTINGS', silent=True)


####################
#  VIEW FUNCTIONS  #
####################

@app.route('/')


####################
#  VIEW FUNCTIONS  #
####################

@app.route('/', methods=['POST'])
def add_dialogue():
    pass

@app.route('/', methods=['POST'])
def add_utterance():
    pass


###############
#  CLI UTILS  #
###############

@app.cli.command('initdb')
def initdb_command():
    """A CLI command to initialize the database"""
    init_db()
    print("Database initialized")

    return


##################
#  DB UTILITIES  #
##################

def init_db():
    """Initialize the database with the schema"""
    db = get_db()
    with app.open_resource('schema.sql', mode='r') as f:
        db.cursor().executescript(f.read())
    db.commit()
    
    return

def get_db():
    """NB: g is the current application context"""
    if not hasattr(g, 'sqlite_db'):
        g.sqlite_db = connect_db()
    return g.sqlite_db

def connect_db():
    """Connect to the specified databse"""
    db = sqlite3.connect(app.config['DATABASE'])
    db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_db(error):
    """Closes the database at the end of the request"""
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()

    return
