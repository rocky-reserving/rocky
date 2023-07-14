from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS, cross_origin

import sys

sys.path.append("../..")

from ..rockycore import ROCKY

app = Flask(__name__)
CORS(app, support_credentials=True)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///instance/app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)


# create a model to store the triangle data
class ROCKYState(db.Model):
    """
    Model to store the state of the ROCKY model
    """

    id = db.Column(db.String, primary_key=True)
    value = db.Column(db.String, default="")

    def __repr__(self):
        return f"<ROCKYState {self.id}>"

@app.route("/api/tri/utils/set-id/<string:id>", methods=["POST"])
def set_id(id):
    




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5432)
