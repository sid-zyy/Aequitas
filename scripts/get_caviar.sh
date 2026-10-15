#!/usr/bin/env sh
# Downloads the 11 CAVIAR clips used for evaluation (~112 MB) into data/videos/caviar.
# Licence: Creative Commons BY-SA. Credit: EC Funded CAVIAR project/IST 2001 37540.
set -e
d="data/videos/caviar"
mkdir -p "$d"
for c in Fight_OneManDown Fight_RunAway1 Fight_Chase Rest_FallOnFloor Rest_SlumpOnFloor Rest_InChair \
         LeftBag LeftBag_PickedUp Browse_WhileWaiting1 Walk1 Meet_Crowd; do
  [ -f "$d/$c.mpg" ] || curl -fL -o "$d/$c.mpg" "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/$c/$c.mpg"
done
echo "Done. Run: python -m aequitas --settings config/eval_caviar.yaml --cameras config/caviar_cameras.yaml run"
