competition/scoring_program.zip: scoring_program/*
	cd scoring_program && zip ./scoring_program.zip *.py && cd .. && mv scoring_program/scoring_program.zip ./competition/

competition.zip: competition/* competition/scoring_program.zip
	cd competition && zip ../competition.zip * && cd ..

submission.zip: submission/*
	cd submission && zip ../submission.zip * && cd ..
