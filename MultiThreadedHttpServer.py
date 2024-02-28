from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler 
import base64, json
from entity import Entity
class RequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        data = {}
        response = ""
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()

        if self.path == "/":
            response = json.dumps({'status': "True"})

        elif self.path == "/score":                     
            data['center'] = Entity.center_coordinates
            data['score'] = Entity.score

            # file = open('shotTarget.jpg', 'rb')
            # img = file.read()
            # data['img'] = base64.b64encode(img).decode('utf-8')

            response = json.dumps(data)

        elif self.path == "/prepare":
            Entity.flag = True  
            response = json.dumps({'status': "True"})
            

        self.wfile.write(bytes(response, 'utf-8'))

    

class Server(ThreadingHTTPServer):
    pass