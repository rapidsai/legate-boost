#!/bin/bash

# adopted from https://github.com/rapidsai/gha-tools/blob/main/tools/rapids-upload-to-anaconda, with some exceptions:
#
#  * assumes artifacts are on GitHub Actions artifact store, not Amazon S3
#  * assumes packages have been unpacked to env variable RAPIDS_LOCAL_CONDA_CHANNEL
#  * does not differentiate between pull request and nightly branches
#    (relies on workflow triggers to just not run this script when it isn't needed)

# publish to the 'experimental' label on the 'legate' channel, for all cases except
# releases (builds triggered by pushing a tag matching this regex exactly, e.g. 'v24.09.00')
if [[ "${GITHUB_REF}" =~ ^refs/tags/v[0-9]+\.[0-9]+\.[0-9]+$ ]];
    declare -r conda_label="main"
else
    declare -r conda_label="experimental"
fi

PKGS_TO_UPLOAD=$(rapids-find-anaconda-uploads.py "${RAPIDS_LOCAL_CONDA_CHANNEL}")

if [ -z "${PKGS_TO_UPLOAD}" ]; then
    rapids-echo-stderr "Couldn't find any packages to upload in: ${RAPIDS_LOCAL_CONDA_CHANNEL}"
    ls -l "${RAPIDS_LOCAL_CONDA_CHANNEL}/"*
    continue
fi

rapids-echo-stderr "Uploading packages to Anaconda.org (channel='legate', label='${conda_label}'): ${PKGS_TO_UPLOAD}"

# export RAPIDS_RETRY_SLEEP=180
# # shellcheck disable=SC2086
# rapids-retry anaconda \
#     -t "${CONDA_LEGATE_TOKEN}" \
#     upload \
#     --label "${RAPIDS_CONDA_UPLOAD_LABEL:-main}" \
#     --skip-existing \
#     --no-progress \
#     ${PKGS_TO_UPLOAD}
