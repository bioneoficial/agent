import unittest
import json
from app import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_home(self):
        response = self.app.get('/')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)