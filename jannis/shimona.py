from matchms.importing import load_from_mgf


spectrum_file_path="./20241003_enamdisc_neg_ms2.mgf"
spectrum = next(load_from_mgf(spectrum_file_path))

print("inchikey:", spectrum.get("inchikey"))
print("inchiaux:", spectrum.get("inchiaux"))
print("inchi:", spectrum.get("inchi"))