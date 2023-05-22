from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS

from .rockycore import ROCKY

app = Flask(__name__)
CORS(app)  # enable

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
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


@app.route(
    "/rockyapi/get-triangle",
    methods=["POST"],
    defaults={"id": None},
    strict_slashes=False,
    endpoint="get_triangle",
    provide_automatic_options=True,
)
def get_triangle(id=None):
    """
    API endpoint to load triangle data stored in the database. If the triangle

    """
    user_id = request.get_json()["user_id"]

    # get the instance data from the database
    rocky_state = ROCKYState.query.get(user_id)

    # if the instance data is not in the database, create it
    if rocky_state is None:
        rocky_state = ROCKYState(id=user_id)
        db.session.add(rocky_state)
        db.session.commit()

    # update the instance state
    temp_rocky = ROCKY()
    temp_rocky.value = rocky_state.value
    temp_rocky.load_taylor_ashe()
    result = temp_rocky.t.paid_loss.tri.to_json()
    print("get_triangle() result:\n", result)

    response = jsonify({"message": "success", "result": result})
    return response


@app.route("/rockyapi/load-taylor-ashe", methods=["POST"])
def load_taylor_ashe():
    user_id = request.get_json()["user_id"]

    # get the instance data from the database
    rocky_state = ROCKYState.query.get(user_id)

    # if the instance data is not in the database, create it
    if rocky_state is None:
        rocky_state = ROCKYState(id=user_id)
        db.session.add(rocky_state)
        db.session.commit()

    # update the instance state
    temp_rocky = ROCKY()
    temp_rocky.value = rocky_state.value
    temp_rocky.load_taylor_ashe()
    result = temp_rocky.t.paid_loss.tri.to_json()
    # result = {"result": "test result"}
    print("load_taylor_ashe() result:", result)

    # save updated instance state to the database
    rocky_state.value = temp_rocky.value
    db.session.commit()

    response = jsonify({"result": result})
    print("JSON response:", response)

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1234)
