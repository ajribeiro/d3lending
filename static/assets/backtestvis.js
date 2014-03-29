jQuery('#myform').submit(function(){
    console.log('submitted')
    $('input[type=submit]').attr('disabled',true);
    loadhtml = "<div id='load' class='loadingdiv'><img src='static/ajax-loader.gif'><p>Recalculating...</p></div>"
    $('body').append(loadhtml)
    console.log(loadhtml)
    $.ajax({
        url : $(this).attr('action'),
        type: $(this).attr('method'),
        dataType: 'json',
        data: $(this).serialize(), 
        success: function(){ 
            plot_backtest() 
        }
    });
    $('button[type=submit], input[type=submit]').attr('disabled',false);
    return false;
});

var colors = ['blue','purple','orange','yellow','cyan','magenta']
var xmargin = 150;
var ymargin = 50;
var aspect = 9./5.
var wbase = document.getElementById('plotarea').offsetWidth
var width = wbase - xmargin;
var height = width/aspect;
var time_extent,time_scale,time_axis
var prof_extent,prof_scale,prof_axis
var ptype = 0

//function to donvert date strings to JS date objects
function to_date(date_str){
    var m = date_str.slice(5,7)
    if(m[0] == '0'){
        m = m.slice(1,2)
    }
    return new Date(date_str.slice(0,4),parseInt(m)-1,date_str.slice(8,10))
}

function sanitize_model_name(name){
    return name.replace(/ /g, '');
}

function get_yval(d){
    if(ptype == 0)
        return d.ret-d.inv
    else if(ptype == 1)
        return d.ret/d.inv
}

function populate_menus(){
    var node = document.createElement("select")
    for(i=2008;i<2012;i++){
        var op = new Option()
        op.value = i
        op.text = i.toString()
        node.options.add(op)
    }
    document.getElementById('syear').appendChild(node)

    node = document.createElement("select")
    for(i=2008;i<2012;i++){
        var op = new Option()
        op.value = i
        op.text = i.toString()
        node.options.add(op)
    }
    document.getElementById('eyear').appendChild(node)

    node = document.createElement("select")
    for(i=1;i<13;i++){
        var op = new Option()
        op.value = i
        op.text = i.toString()
        node.options.add(op)
    }
    document.getElementById('smon').appendChild(node)

    node = document.createElement("select")
    for(i=1;i<13;i++){
        var op = new Option()
        op.value = i
        op.text = i.toString()
        node.options.add(op)
    }
    document.getElementById('emon').appendChild(node)

    node = document.createElement("select")
    for(i=1;i<32;i++){
        var op = new Option()
        op.value = i
        op.text = i.toString()
        node.options.add(op)
    }
    document.getElementById('sday').appendChild(node)

    node = document.createElement("select")
    for(i=1;i<32;i++){
        var op = new Option()
        op.value = i
        op.text = i.toString()
        node.options.add(op)
    }
    document.getElementById('eday').appendChild(node)
}

function create_scales(data){

    time_extent = []
    for(var key in data){
        //find the range for the time scale, make the scale
        var ext = d3.extent(data[key],function(d){
            return to_date(d.date)
        });
        if(time_extent.length == 0)
            time_extent = ext
        else{
            if(ext[0] < time_extent[0])
                time_extent[0] = ext[0]
            if(ext[1] > time_extent[1])
                time_extent[1] = ext[1]
        }
    }

    time_scale = d3.time.scale.utc()
        .domain(time_extent)
        .range([xmargin, width]);


    prof_extent = []
    for(var key in data){
        //find the range for the profit scale, make the scale
        var ext = d3.extent(data[key],function(d){
            if(ptype == 0)
                return d.ret-d.inv
            else if(ptype == 1)
                return d.ret/d.inv
        })
        if(prof_extent.length == 0)
            prof_extent = ext
        else{
            if(ext[0] < prof_extent[0])
                prof_extent[0] = ext[0]
            if(ext[1] > prof_extent[1])
                prof_extent[1] = ext[1]
        }
    }

    prof_scale = d3.scale.linear()
        .domain(prof_extent)
        .range([height+ymargin,ymargin])
}

function add_label(circle, d, type){

    d3.select(circle)
        .transition()
        .attr("r", 9);

    d3.select('#linechart')
        .append("text")
        .text(type.slice(0,1))
        .attr("x", function(){
            return time_scale(to_date(d['date']))
        })
        .attr("y", function(){
            return prof_scale(get_yval(d))
        })
        .attr("dy", "0.35em")
        .attr("class","linelabel "+type)
        .attr("text-anchor","middle")
        .style("opacity", 0)
        .style("fill", "white")
        .transition()
        .style("opacity", 1);
}

function draw_axes(){

    //create the time axis
    time_axis = d3.svg.axis()
        .scale(time_scale);

    //draw the time axis
    d3.select("svg")
        .append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0,"+(height+ymargin)+")")
        .call(time_axis);

    d3.select('#linechart')
        .append('text')
        .attr('class','axis_label')
        .text('Listing Date')
        .attr('x',xmargin + width/2.)
        .attr('y',ymargin+height+50)

    //create the profit axis
    var prof_axis = d3.svg.axis()
        .scale(prof_scale)
        .orient("left");

    //draw the time axis
    d3.select("svg")
        .append("g")
        .attr("class", "y axis")
        .attr("transform", "translate(" + xmargin + ",0)")
        .call(prof_axis);
}

function plot_backtest(){

    var dd = {}

    dd["syear"] = parseInt($("#syear")[0].childNodes[0].value)
    dd["smon"] = parseInt($("#smon")[0].childNodes[0].value)
    dd["sday"] = parseInt($("#sday")[0].childNodes[0].value)

    dd["eyear"] = parseInt($("#eyear")[0].childNodes[0].value)
    dd["emon"] = parseInt($("#emon")[0].childNodes[0].value)
    dd["eday"] = parseInt($("#eday")[0].childNodes[0].value)


    var cks = document.getElementsByName('mtype');
    var models = []
    for(var i=0; i<cks.length; i++){
        if(cks[i].checked){
            models.push(cks[i].value)
        }
    }

    if(models.length < 1){
        alert('Please select at least one model')
        return
    } 

    dd['models'] = models


    $.ajax({
        type: 'POST',
        url: '_do_backtest',
        contentType: 'application/json',
        data: JSON.stringify(dd),
        dataType: 'json',
        success: function(result){
            $('#load').remove()
            data = result
            wbase = document.getElementById('plotarea').offsetWidth
            width = wbase - xmargin;
            height = width/aspect;

            var radios = document.getElementsByName('ptype');
            for(var i=0; i<radios.length; i++){
                if(radios[i].checked){
                    ptype = i
                    break
                }
            }

            $('#linechart').remove()
            //create the chart
            var chart = d3.select("#plotarea")
                .append("svg")
                .attr("width", width+xmargin*2)
                .attr("height", height+ymargin*2)
                .attr('x',xmargin)
                .attr('y',ymargin)
                .attr('id','linechart')

            create_scales(data)
            draw_axes()

            rad = 1
            for(var key in result){
                draw(result[key],key,chart,rad)
                rad += 1
            }

        },
        error: function(emsg){
            console.log(emsg)
        }
    });
}

function set_mouseover_actions(chart,type,data){

    chart.selectAll("circle."+sanitize_model_name(type))
        .on("mouseover", function(d){
            d3.select(this)
            .transition()
            .attr("r",9);
        })
        .on("mouseout", function(d,i){
            if(i !== data.length-1){
                d3.select(this)
                    .transition()
                    .attr("r", 5);
            }
        });


    chart.selectAll("circle."+sanitize_model_name(type))
        .on("mouseover.tooltip", function(d){
            d3.select('#dateproftool').remove()
            d3.select('#ptypetool').remove()
            d3.select('#tipbg').remove()

            chart.append('rect')
                .attr("x", function(){
                    return time_scale(to_date(d.date)) + 10
                })
                .attr("y", function(){
                    return prof_scale(get_yval(d,ptype)) - 60
                })
                .attr('id','tipbg')
                .attr('class','tipbg')
                .attr('height','60px')
                .attr('width',function(){
                    if(ptype == 0)
                        return 210 + Math.round(get_yval(d)).toString().length*10
                    else
                        return '230px'
                })

            chart.append("text")
                .text(function(){
                    var rtxt = 'Return:  '
                    if(ptype == 0)
                        rtxt += '$'+Math.round(get_yval(d)).toString()
                    else if(ptype == 1){
                        rtxt += get_yval(d).toString().slice(0,4)+'%'
                    }
                    return rtxt
                })
                .attr("x", function(){
                    return time_scale(to_date(d.date)) + 15
                })
                .attr("y", function(){
                    return prof_scale(get_yval(d)) - 10
                })
                .attr("id", 'dateproftool')
                .attr('class','labeltip')

            chart.append("text")
                .text(function(){
                    return type+' '+d.date
                })
                .attr("x", time_scale(to_date(d.date)) + 15)
                .attr("y", function(){
                    return prof_scale(get_yval(d)) - 35
                })
                .attr("id", 'ptypetool')
                .attr('class','labeltip')

            var vline,hline
            if(d3.select('#vertline').empty()){
                vline = chart.append('line')
                    .attr('id','vertline')
                    .attr('class','crosshatch')
            }
            else{
                vline = d3.select('#vertline')
            }

            vline.attr('x1',function(){
                    return time_scale(to_date(d.date))
                })
                .attr('x2',function(){
                    return time_scale(to_date(d.date))
                })
                .attr('y1',function(){
                    return prof_scale(prof_extent[0])
                })
                .attr('y2',function(){
                    return prof_scale(get_yval(d))
                })
            

            if(d3.select('#horline').empty()){
                hline = chart.append('line')
                    .attr('id','horline')
                    .attr('class','crosshatch')
            }
            else{
                hline = d3.select('horline')
            }
            hline.attr('x1',function(){
                    return time_scale(time_extent[0])
                })
                .attr('x2',function(){
                    return time_scale(to_date(d.date))
                })
                .attr('y1',function(){
                    return prof_scale(get_yval(d))
                })
                .attr('y2',function(){
                    return prof_scale(get_yval(d))
                })

        });


    chart.selectAll("circle")
        .on("mouseout.tooltip", function(d){
            d3.select('#dateproftool').remove()
            d3.select('#ptypetool').remove()
            d3.select('#vertline').remove()
            d3.select('#horline').remove()
            d3.select('#tipbg').remove()
        });
}

//function to plot the trading data
function draw(data,type,chart,rad){

    //define the function which will draw the line
    var lineFunction = d3.svg.line()
        .x(function(d){ 
            return time_scale(to_date(d['date']))
        })
        .y(function(d){
            return prof_scale(get_yval(d))
        })
        .interpolate("linear");

    var lineGraph = chart.append("path")
        .attr('id','profline'+type)
        .attr("d", lineFunction(data))
        .attr('class',sanitize_model_name(type))
        // .style("stroke-dasharray", function(){
        //     return ((5).toString()+","+(5).toString())
        // })

    //plot the points
    d3.select("#linechart")
        .selectAll("circle."+sanitize_model_name(type))
        .data(data)
        .enter()
        .append("circle")
        .attr('cx',function(d){
            return time_scale(to_date(d['date']))
        })
        .attr('cy',function(d){
            return prof_scale(get_yval(d))
        })
        .attr('class',sanitize_model_name(type))
        .attr('r',0)

    var enter_duration = 5000;

    d3.select("#linechart")
        .selectAll("circle."+sanitize_model_name(type))
        .transition()
        .delay(function(d, i){
            return i / data.length * enter_duration;
        })
        .attr("r", 5)
        .each("end",function(d,i){
            if(i == data.length-1){
                add_label(this,d,type);
            }
        });

    set_mouseover_actions(chart,type,data)


}

