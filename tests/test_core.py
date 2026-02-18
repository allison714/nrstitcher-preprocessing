import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import map_index, tile_idx_to_xy, xy_to_tile_idx, parse_filename, ScanOrder

class TestCoreLogic(unittest.TestCase):
    
    def test_map_index(self):
        # i, n_channels, z_slices -> tile, z, ch
        # C=2, Z=3
        # Tile 0:
        #   Z0: C0(0), C1(1)
        #   Z1: C0(2), C1(3)
        #   Z2: C0(4), C1(5)
        # Tile 1:
        #   Z0: C0(6)...
        
        nc = 2
        nz = 3
        
        self.assertEqual(map_index(0, nc, nz), (0, 0, 0))
        self.assertEqual(map_index(1, nc, nz), (0, 0, 1))
        self.assertEqual(map_index(2, nc, nz), (0, 1, 0))
        self.assertEqual(map_index(5, nc, nz), (0, 2, 1))
        self.assertEqual(map_index(6, nc, nz), (1, 0, 0))
        
    def test_tile_idx_to_xy_col_serpentine(self):
        # Column Serpentine (pan-ASLM): X slow, Y fast
        # n_tiles_x = 3, n_tiles_y = 3
        # Col 0 (x=0, even): 0->(0,0), 1->(0,1), 2->(0,2)
        # Col 1 (x=1, odd):  3->(1,2), 4->(1,1), 5->(1,0)
        # Col 2 (x=2, even): 6->(2,0), 7->(2,1), 8->(2,2)
        
        nx, ny = 3, 3
        order = ScanOrder.COL_SERPENTINE.value
        
        # Col 0
        self.assertEqual(tile_idx_to_xy(0, nx, ny, order), (0, 0))
        self.assertEqual(tile_idx_to_xy(1, nx, ny, order), (0, 1))
        self.assertEqual(tile_idx_to_xy(2, nx, ny, order), (0, 2))
        
        # Col 1 (reversed)
        self.assertEqual(tile_idx_to_xy(3, nx, ny, order), (1, 2))
        self.assertEqual(tile_idx_to_xy(4, nx, ny, order), (1, 1))
        self.assertEqual(tile_idx_to_xy(5, nx, ny, order), (1, 0))
        
        # Col 2
        self.assertEqual(tile_idx_to_xy(6, nx, ny, order), (2, 0))
        
    def test_tile_idx_to_xy_row_serpentine(self):
        # Row Serpentine: Y slow, X fast
        nx, ny = 3, 3
        order = ScanOrder.ROW_SERPENTINE.value
        
        # Row 0 (even): left to right
        self.assertEqual(tile_idx_to_xy(0, nx, ny, order), (0, 0))
        self.assertEqual(tile_idx_to_xy(1, nx, ny, order), (1, 0))
        self.assertEqual(tile_idx_to_xy(2, nx, ny, order), (2, 0))
        
        # Row 1 (odd): right to left
        self.assertEqual(tile_idx_to_xy(3, nx, ny, order), (2, 1))
        self.assertEqual(tile_idx_to_xy(4, nx, ny, order), (1, 1))
        self.assertEqual(tile_idx_to_xy(5, nx, ny, order), (0, 1))
    
    def test_xy_to_tile_roundtrip(self):
        # Verify roundtrip: idx -> xy -> idx for all orders
        nx, ny = 4, 3
        for order in [ScanOrder.COL_SERPENTINE.value, ScanOrder.ROW_SERPENTINE.value, ScanOrder.RASTER.value]:
            for t in range(nx * ny):
                x, y = tile_idx_to_xy(t, nx, ny, order)
                self.assertEqual(xy_to_tile_idx(x, y, nx, ny, order), t, 
                    f"Roundtrip failed for order={order}, t={t}, xy=({x},{y})")
        
    def test_parse_filename(self):
        self.assertEqual(parse_filename("img_0001.tif"), 1)
        self.assertEqual(parse_filename("img_0100.tiff"), 100)
        self.assertEqual(parse_filename("img_1.tif"), 1)
        self.assertEqual(parse_filename("experiment_s01_t005.tif"), 5)
        self.assertEqual(parse_filename("no_digits.tif"), None)
        self.assertEqual(parse_filename("digits_in_middle_123_text.tif"), None)
        
if __name__ == '__main__':
    unittest.main()
