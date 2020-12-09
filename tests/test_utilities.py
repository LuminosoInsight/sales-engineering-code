import unittest

class TestUtilityFunctions(unittest.TestCase):

    def test_api_url1(self):
        self.url1 = "https://analytics.luminoso.com/app/projects/p87t862f/prt5cx7f"
        self.api_url, self.from_proj = parse_url(self.url1)
        self.assertEqual(self.api_url,"https://analytics.luminoso.com/api/v5/")
        self.assertEqual(self.from_proj,"prt5cx7f")

    def test_url2(self):
        self.url2 = "https://analytics.luminoso.com/app/projects/p87t862f/prt5cx7f/highlights"
        self.api_url, self.from_proj = parse_url(self.url2)
        self.assertEqual(self.api_url,"https://analytics.luminoso.com/api/v5/")
        self.assertEqual(self.from_proj,"prt5cx7f")

    def test_url3(self):
        self.url3 = "https://analytics.luminoso.com/app/projects/p87t862f/prt5cx7f/galaxy?suggesting=true"
        self.api_url, self.from_proj = parse_url(self.url3)
        self.assertEqual(self.api_url,"https://analytics.luminoso.com/api/v5/")
        self.assertEqual(self.from_proj,"prt5cx7f")
        
    def test_url4(self):
        self.url4 = "https://analytics.luminoso.com/app/projects/p87t862f/prt5cx7f/galaxy?suggesting=false&search=cool"
        self.api_url, self.from_proj = parse_url(self.url4)
        self.assertEqual(self.api_url,"https://analytics.luminoso.com/api/v5/")
        self.assertEqual(self.from_proj,"prt5cx7f")

if __name__ == '__main__':
    unittest.main()