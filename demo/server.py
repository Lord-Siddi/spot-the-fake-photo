import os
import sys
import json
import numpy as np
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer

# Configure paths to allow importing predict.py from the workspace root
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_dir)
sys.path.append(workspace_root)

from predict import predict

class DemoServerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress logging to keep output stream clean
        return

    def do_GET(self):
        # Handle index.html requests
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            index_path = os.path.join(current_dir, 'index.html')
            with open(index_path, 'r', encoding='utf-8') as f:
                self.wfile.write(f.read().encode('utf-8'))
        else:
            self.send_error(404, 'File Not Found')

    def do_POST(self):
        # Handle prediction requests
        if self.path == '/predict':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length == 0:
                    self.send_json_response({"error": "Empty payload"}, 400)
                    return
                
                post_data = self.rfile.read(content_length)
                
                # Write received JPEG frame temporarily to disk
                temp_img_path = os.path.join(current_dir, 'temp_demo_frame.jpg')
                with open(temp_img_path, 'wb') as f:
                    f.write(post_data)
                
                # Execute prediction pipeline
                score = predict(temp_img_path)
                
                # Clean up the temporary file immediately
                try:
                    os.remove(temp_img_path)
                except:
                    pass
                
                # Send prediction score back as json response
                self.send_json_response({"score": score})
                
            except Exception as e:
                self.send_json_response({"error": str(e)}, 500)
        else:
            self.send_error(404, 'Endpoint Not Found')

    def send_json_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

def run_server(port=None):
    if port is None:
        port = int(os.environ.get('PORT', 5000))
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, DemoServerHandler)
    print(f"\n=======================================================")
    print(f"  Live Demo Server is running successfully!")
    print(f"  Open http://localhost:{port}/ in your web browser.")
    print(f"  Press Ctrl+C to terminate.")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo server...")
        httpd.server_close()

if __name__ == '__main__':
    run_server()
