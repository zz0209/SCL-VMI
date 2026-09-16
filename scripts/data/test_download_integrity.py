"""Small offline checks for hash semantics and fail-closed manifest handling."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from download_flare import check_inventory, file_digest, safe_file

class IntegrityTests(unittest.TestCase):
    def test_git_blob_and_lfs_digests(self):
        payload = b'medical-volume-fixture\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'sample'
            path.write_bytes(payload)
            for entry in [
                {'size':len(payload),'oid':hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()},
                {'size':len(payload),'lfs':{'oid':hashlib.sha256(payload).hexdigest()}},
            ]:
                actual, expected = file_digest(path, entry)
                self.assertEqual(actual, expected)
                path.write_bytes(b'x'+payload[1:])
                self.assertNotEqual(file_digest(path, entry)[0], expected)
                path.write_bytes(payload)

    def test_masked_hash_rejected(self):
        public = {'repo_id':'test','revision':'abc','files':[{'path':'a','size':1}]}
        private = {**public,'files':[{'path':'a','size':1,'lfs':{'oid':'*'*64}}]}
        with self.assertRaises(ValueError): check_inventory(public, private)
        private['files'][0]['lfs']['oid'] = 'a'*64
        check_inventory(public, private)

    def test_changed_tree_and_duplicate_rejected(self):
        public = {'repo_id':'test','revision':'abc','files':[{'path':'a','size':1}]}
        private = {**public,'files':[{'path':'b','size':1,'oid':'a'*40}]}
        with self.assertRaises(ValueError): check_inventory(public, private)
        private['files'] = [{'path':'a','size':1,'oid':'a'*40}]*2
        with self.assertRaises(ValueError): check_inventory(public, private)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.assertEqual(safe_file(root,'inside/file'),(root/'inside/file').resolve())
            with self.assertRaises(ValueError): safe_file(root,'../outside')

if __name__ == '__main__': unittest.main()
