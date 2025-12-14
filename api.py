from flask import Flask, request
from flask_restful import Resource, Api
import os

app = Flask(__name__)
api = Api(app)

class dataIngestor(Resource):
    # def get(self):
    #     return {
    #             'message': 'This endpoint is for storing data',
    #             'documentation':'Method for sending data: POST\nExample payload is mentioned in the schema',
    #             'schema': ingest.TEMPLATE_PAYLOAD
    #         }
    
    def post(self, orderId):
        # request.json
        data:dict = request.get_json()
        
        return {}
    
api.add_resource(dataIngestor, '/ingestion/<int:orderId>')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))