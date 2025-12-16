rm -f python_code.zip
zip -r python_code.zip . -i '*.py' -x 'submissions/*'
echo "python_code.zip was created"
