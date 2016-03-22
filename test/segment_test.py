# -*- coding: utf-8 -*-
import unittest
from prep import morph


class TestSegment(unittest.TestCase):
    def test_segment(self):
        self.assertEqual(morph.segment_hashtag("#suchalovelyday"), " hashtag such a lovely day . ")
        self.assertEqual(morph.segment_hashtag("#lovesnails"), " hashtag love snails . ")
        self.assertEqual(morph.segment_hashtag("#therapistfinder"), " hashtag therapist finder . ")
        self.assertEqual(morph.segment_hashtag("#whorepresents"), " hashtag who represents . ")
        self.assertEqual(morph.segment_hashtag("#zugarrivesatgaredunord"), " hashtag zug arrives at gare du nord . ")

    def test_ends_with_a_label(self):
        self.assertEqual(morph.ends_with_label("Ziggurat,Solomon\n"), 1)
        self.assertEqual(morph.ends_with_label("Ziggurat,\n"), 0)
        self.assertEqual(morph.ends_with_label("Ziggurat\n"), -1)
        self.assertEqual(morph.ends_with_label("Dear Coffee ☕️,"), 0)


if __name__ == '__main__':
    unittest.main()
