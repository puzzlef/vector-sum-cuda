const fs = require('fs');
const os = require('os');
const path = require('path');

const RELEMS = /^# Elements (.+)/m;
const RRESLT = /^\[(.+?) ms\] \[(.+?)\] (\w+)(?:<<<auto, (\d+)>>> \[thread-duty=(\d+)\])?/m;
const RPSUMM = /^(.+?),(.+?),(.+?),(.+?),"(.+?)",(.+?),(.+?),(.+)/m;
const RPTECH = /^sumCuda/m;




// *-FILE
// ------

function readFile(pth) {
  var d = fs.readFileSync(pth, 'utf8');
  return d.replace(/\r?\n/g, '\n');
}

function writeFile(pth, d) {
  d = d.replace(/\r?\n/g, os.EOL);
  fs.writeFileSync(pth, d);
}




// *-CSV
// -----

function writeCsv(pth, rows) {
  var cols = Object.keys(rows[0]);
  var a = cols.join()+'\n';
  for (var r of rows)
    a += [...Object.values(r)].map(v => `"${v}"`).join()+'\n';
  writeFile(pth, a);
}




// *-LOG
// -----

function readLogLine(ln, data, state) {
  if (RELEMS.test(ln)) {
    var [, elements] = RELEMS.exec(ln);
    var elements = parseFloat(elements);
    if (!data.has(elements)) data.set(elements, []);
    state = {elements};
  }
  else if (RRESLT.test(ln)) {
    var [, time, sum, technique, block_size, thread_duty] = RRESLT.exec(ln);
    data.get(state.elements).push(Object.assign({}, state, {
      time:        parseFloat(time),
      sum:         parseFloat(sum),
      technique,
      block_size:  parseFloat(block_size||'0'),
      thread_duty: parseFloat(thread_duty||'0')
    }));
  }
  return state;
}

function readLog(pth) {
  var text = readFile(pth);
  var lines = text.split('\n');
  var data = new Map();
  var state = null;
  for (var ln of lines)
    state = readLogLine(ln, data, state);
  return data;
}




// *-PROF
// ------

function readProfLine(ln, data, state) {
  if (RPSUMM.test(ln)) {
    var [, id, time, api_call_id, function_name, demangled_name,
      device_name, sol_sm_percent, sol_memory_percent] = RPSUMM.exec(ln);
    if (state==null) {
      var group = 'all';
      if (!data.has(group)) data.set(group, []);
      state = {group};
    }
    if (id!=='ID') data.get(state.group).push(Object.assign({}, state, {
      id:                 parseFloat(id),
      time,
      api_call_id:        parseFloat(api_call_id),
      function_name,
      demangled_name,
      device_name,
      sol_sm_percent:     parseFloat(sol_sm_percent),
      sol_memory_percent: parseFloat(sol_memory_percent),
    }));
  }
  return state;
}

function profFields(r) {
  var sol_sm_percent     = r? r.sol_sm_percent : 0;
  var sol_memory_percent = r? r.sol_memory_percent : 0;
  return {sol_sm_percent, sol_memory_percent};
}

function readProf(pth) {
  var text = readFile(pth);
  var lines = text.split('\n');
  var data = new Map();
  var state = null;
  for (var ln of lines)
    state = readProfLine(ln, data, state);
  return data;
}




// PROCESS-*
// ---------

function processCsv(data) {
  var a = [];
  for (var rows of data.values())
    a.push(...rows);
  return a;
}


function processProf(data, prof) {
  var prows = prof.get('all'), i = 0;
  for (var rows of data.values()) {
    for (var r of rows) {
      var pr = RPTECH.test(r.technique)? prows[i++] : null;
      Object.assign(r, profFields(pr));
    }
  }
  return data;
}




// MAIN
// ----

function readInput(inp) {
  var [log, prof] = inp.split('+');
  var data = readLog(log);
  if (prof) processProf(data, readProf(prof));
  return data;
}

function main(cmd, inp, out) {
  var data = readInput(inp);
  if (path.extname(out)==='') cmd += '-dir';
  switch (cmd) {
    case 'csv':
      var rows = processCsv(data);
      writeCsv(out, rows);
      break;
    case 'csv-dir':
      for (var [graph, rows] of data)
        writeCsv(path.join(out, graph+'.csv'), rows);
      break;
    default:
      console.error(`error: "${cmd}"?`);
      break;
  }
}
main(...process.argv.slice(2));
