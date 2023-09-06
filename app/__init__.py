# This will initialize the Flask app.

from flask import Flask
import os
import logging

def create_app():
    app = Flask(__name__)

    # Configure logging
    log_folder = os.path.join(app.root_path, 'logs')
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, 'app.log')

    logging.basicConfig(filename=log_file, level=logging.DEBUG)

    from . import routes
    app.register_blueprint(routes.bp)

    return app
