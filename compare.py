import face_recognition as fr;import os;import sys;import glob
def run():
 try:
  x=['jpeg','png','JPEG','PNG'];fa=None;fb=None
  for e in x:
   if os.path.exists(f'A.{e}'): fa=f'A.{e}'
   if os.path.exists(f'B.{e}'): fb=f'B.{e}'
  if not fa or not fb: print("Error:A.* or B.* not found");sys.exit(1)
  print(f"Comparing {fa} and {fb}...");ia=fr.load_image_file(fa);ib=fr.load_image_file(fb);ea=fr.face_encodings(ia);eb=fr.face_encodings(ib)
  if len(ea)==0 or len(eb)==0: print("Error:No face detected in one of the images");sys.exit(1)
  dist=fr.face_distance([ea[0]],eb[0])[0];score=max(0,min(100,(1.0-dist)*100));print(f"Similarity: {score:.2f}%");os.system(f'echo "similarity={score:.2f}" >> $GITHUB_OUTPUT')
 except Exception as e: print(f"Fatal Error:{e}");sys.exit(1)
if __name__=="__main__": run()
