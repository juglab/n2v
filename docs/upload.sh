#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUTDIR="$DIR/build/html"

FTP_HOST=home18440447.1and1-data.host
FTP_USER=p6831244-fjug
FTP_TARGET_DIR=/doc

lftp sftp://$FTP_USER@$FTP_HOST -e "set sftp:auto-confirm yes ; mirror -R $OUTPUTDIR $FTP_TARGET_DIR ; quit"
