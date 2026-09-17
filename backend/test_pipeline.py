from pipeline import analyze_patient


result = analyze_patient(
    "dataset/HVSMR/cropped/cropped/pat17_cropped.nii.gz",
    "pat17_backend_test"
)

print("\n========================================")
print("PIPELINE TEST SUCCESSFUL")
print("========================================")