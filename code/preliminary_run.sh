# !/bin/bash

TEST_CASES_DIR=${1}
RES_DIR=${2}

mkdir -p ${RES_DIR}/logs

cat ${TEST_CASES_DIR}/gt.txt | parallel \
--jobs 1 \
--timeout 120 \
--colsep ',' \
--joblog ${RES_DIR}/joblog.txt \
--results ${RES_DIR}/logs \
"python verifier.py --net {1} --spec ${TEST_CASES_DIR}/{1}/{2}"
