"""
Prepare the clusters in result/ for annotation (run with PyCharm's Run
button). This script only prepares and suggests - it never labels or
changes a cluster:

  1. imports result/clusters.csv (first run only)
  2. names every cluster automatically with TrOCR (auto_name column)
  3. writes review sheets: flags, each word's reading, automatic names
  4. prints the clusters with their automatic names

Then the annotator decides each cluster from the command line, e.g.
    label 9 concept                -> keep the automatic name
    label 9 concept "Padova"       -> type the name instead
    remove 7                       -> reject the cluster
(see the commands printed at the end).

Run it again after moving words: names are updated, and only words not read
before go through TrOCR. Delete result/annotations/ to start from scratch.
"""

from pathlib import Path

from core.comparing_all_similar_words import review
from annotator import recognition
from annotator.annotation import ClusterStore

HERE = Path(__file__).resolve().parent
RESULT = HERE / "result"                                                  # clusters.csv
IMAGES = Path("/home/isuri/PycharmProjects/SETS_image_processing/data2")   # page images
ANNOTATIONS = RESULT / "annotations"
SHEETS = HERE / "review"

USE_TROCR = True                                    # False: skip automatic names
TROCR_MODEL = "microsoft/trocr-base-handwritten"    # or a fine-tuned local folder
LEXICON = None                                      # optional: HERE / "names.txt"

store = ClusterStore(ANNOTATIONS)

# 1. Import (first run only)
if not store.runs():
    print(f"1. imported run '{store.import_run(RESULT, image_dir=IMAGES)}'")
run = store.runs()[-1]
clusters = store.list_clusters(run)
print(f"   {len(clusters)} clusters, {sum(c['size'] for c in clusters)} words")

# 2. Automatic names
if USE_TROCR:
    try:
        lexicon = recognition.load_lexicon(LEXICON) if LEXICON else None
        res = recognition.auto_name_clusters(
            store, run, recognition.TrOCRRecogniser(TROCR_MODEL), lexicon=lexicon,
            progress=lambda k, t: print(f"   {k}/{t} words read", end="\r", flush=True))
        print(f"\n2. {res['read']} words read; {res['named']} clusters named automatically")
    except ImportError as e:
        print(f"2. automatic names skipped: {e}")

# 3. Review sheets
sheets = review.write_review_sheets(store, run, SHEETS)
print(f"3. review sheets in {SHEETS}: {', '.join(p.name for p in sheets)}")

# 4. Current state
print("4. clusters:")
for c in store.list_clusters(run):
    print(f"   #{c['number']:<4} {c['status']:<9} {c['size']:>3} words  {store.label_text(c)}")
print(f"   progress: {store.progress(run)}")

PROJECT_ROOT = HERE.parents[1]                      # .../SETS_image_processing
A = (f"python -m core.comparing_all_similar_words.annotation "
     f"--dir {ANNOTATIONS.relative_to(PROJECT_ROOT)}")
print(f"""
Annotate from the project root, for example:
   {A} label 9 concept --by isuri             # keep the automatic name
   {A} label 9 concept "Padova" --by isuri    # type the name yourself
   {A} remove 7 --reason noise --by isuri
   {A} move 41 --to 10 --by isuri
   {A} export labels.csv""")