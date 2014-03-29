from flask import Flask, render_template, jsonify,  \
            request, make_response

import json
import backtester
import datetime as dt

app = Flask(__name__)
# app.config.from_object(__name__)


@app.route('/_noop')
def _noop():
    return jsonify(result=1)

@app.route('/_do_backtest',methods=['POST', 'GET'])
def _do_backtest():
    # json.loads(request)
    dd = request.json
    data = backtester.backtester(dt.datetime(dd['syear'],dd['smon'],dd['sday']), \
            dt.datetime(dd['eyear'],dd['emon'],dd['eday']),dd['models'])
    return data

@app.route('/')
def backtestapp():
    return render_template('index.html')

if __name__ == '__main__':
    app.debug=True
    app.run(host='0.0.0.0',port=80)
    
