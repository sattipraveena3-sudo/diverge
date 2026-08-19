from diverge.ctu_chb import adverse_outcome, parse_header


def test_parse_header_accepts_physionet_whitespace_metadata(tmp_path):
    header = tmp_path / "1001.hea"
    header.write_text(
        """1001 2 4 19200
1001.dat 16 100(0)/bpm 12 0 15050 20101 0 FHR
1001.dat 16 100/nd 12 0 700 378 0 UC
#-- Outcome measures
#pH           7.04
#BE           -10.5
#Apgar1       6
#Apgar5       8
#Weight(g)    2660
""",
        encoding="utf-8",
    )

    metadata = parse_header(header)

    assert metadata.record_id == "1001"
    assert metadata.ph == 7.04
    assert metadata.base_excess == -10.5
    assert metadata.apgar1 == 6.0
    assert metadata.apgar5 == 8.0
    assert metadata.birth_weight == 2660.0
    assert adverse_outcome(metadata) == 1
