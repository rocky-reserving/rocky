import os

from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from triangle import Triangle
from rocky import ROCKY

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)


# create a model to store the triangle data
class ROCKYModel(db.Model):
    """
    Model to store the triangle data. The triangle data is stored as a table of 4 columns:
    1. triangle_id: the id of the triangle
    2. accident_id: the id of the accident period
    3. development_id: the id of the development period
    4. value: the value in the (acc, dev) cell of the triangle
    """

    triangle_id = db.Column(db.Integer, primary_key=True)
    accident_id = db.Column(db.Integer, nullable=False)
    development_id = db.Column(db.Integer, nullable=False)
    value = db.Column(db.Float, nullable=False, default=0.0)

    def __repr__(self):
        return f"<ROCKYModel {self.triangle_id}>"
