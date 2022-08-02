---
layout: page
title: Projects
sidebar_link: true
---


<head>

<style>
/* Set height of body and the document to 100% to enable "full page tabs" */
body, html {
  height: 100%;
  margin: 0;
  font-family: Arial;
}

/* Style tab links */
.tablink {
  background-color: #555;
  color: white;
  float: left;
  border: none;
  outline: none;
  cursor: pointer;
  padding: 14px 16px;
  font-size: 17px;
  width: 33%;
}

.tablink:hover {
  background-color: #777;
}

/* Style the tab content (and add height:100% for full page content) */
.tabcontent {
  color: black;
  display: none;
  padding: 100px 20px;
  height: 100%;
}

.content {
    padding-top: 0rem;
    padding-bottom: 4rem;
}

.logo{
  width: 100;
  margin-right: 20px;
}

table, tr, td {
    border: none;
}


</style>


<script>
function openPage(pageName, elmnt) {
  // Hide all elements with class="tabcontent" by default */
  var i, tabcontent, tablinks, color;
  color = '#080450';
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }

  // Remove the background color of all tablinks/buttons
  tablinks = document.getElementsByClassName("tablink");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].style.backgroundColor = "";
  }

  // Show the specific tab content
  document.getElementById(pageName).style.display = "block";

  // Add the specific color to the button used to open the tab content
  elmnt.style.backgroundColor = color;
}

// Get the element with id="defaultOpen" and click on it
document.getElementById("defaultOpen").click();

</script>

</head>

<br/>

<button class="tablink" onclick="openPage('Python', this)" id="defaultOpen">Python</button>
<button class="tablink" onclick="openPage('R', this)">R</button>
<button class="tablink" onclick="openPage('Datasets', this)">Datasets</button>

<div id="R" class="tabcontent">



<table>
<thead>

<tr>
<td><a href="https://krzjoa.github.io/torchts"><img src='https://raw.githubusercontent.com/krzjoa/torchts/master/man/figures/logo-small.png' width="100" style="margin-right: 20px" /></td>  
<td><a href = "https://github.com/krzjoa/torchts"><b>torchts</b></a><br>Time series models with torch</td>
</tr>


<tr>
<td><a href="https://krzjoa.github.io/awesome-r-dataviz"><img src='https://raw.githubusercontent.com/krzjoa/awesome-r-dataviz/master/logo/logo.png' width="100" style="margin-right: 20px" /></td>  
<td><a href = "https://github.com/krzjoa/awesome-r-dataviz"><b>awesome-r-dataviz</b></a><br>Curated resources for Data Visualization in R.</td>
</tr>

<tr>
<td><a href="https://krzjoa.github.io/matricks"><img src='https://raw.githubusercontent.com/krzjoa/matricks/master/man/figures/logo.png' width="100" style="margin-right: 20px" /></td>  
<td><a href = "https://github.com/krzjoa/matricks"><b>matricks</b></a><br>Useful tricks for matrix manipulation</td>
</tr>

<tr>
<td><a href="https://krzjoa.github.io/eponge"><img src='https://raw.githubusercontent.com/krzjoa/eponge/master/man/figures/logo.png' width="100" style="margin-right: 20px" /></td>  
<td><a href = "https://github.com/krzjoa/eponge"><b>eponge</b></a><br>Small utility which makes selective objects removing easier</td>
</tr>

<tr>
<td><a href="https://krzjoa.github.io/path.chain"><img src='https://raw.githubusercontent.com/krzjoa/path.chain/master/man/figures/logo.png' width="100" style="margin-right: 20px" /></td>  
<td><a href = "https://github.com/krzjoa/path.chain"><b>path.chain</b></a> <br>Concise structure for path chaining</td>
</tr>

<tr>
<td><a href="https://krzjoa.github.io/wayfarer"><img src='https://raw.githubusercontent.com/krzjoa/wayfarer/master/man/figures/logo.png' width="100" style="margin-right: 20px" /></td>  
<td><a href = "https://github.com/krzjoa/wayfarer"><b>wayfarer</b></a> <br>Tools for working with Awesome Lists</td>
</tr>

</thead>
</table>

</div>


<div id="Python" class="tabcontent">


<table>
<thead>

<tr>
<td><a href="https://krzjoa.github.io/awesome-python-data-science"><img src='https://raw.githubusercontent.com/krzjoa/awesome-python-data-science/master/img/py-datascience.png' width="100" style="margin-right: 20px" /></td>  
<td><a href = "https://github.com/krzjoa/awesome-python-data-science"><b>awesome-python-data-science</b></a> <br>Curated resources for Data Visualization in R.</td>
</tr>

<tr>
<td><a href="https://github.com/krzjoa/bace"><img src='https://raw.githubusercontent.com/krzjoa/bace/master/img/bace-of-spades.png' width="100" style="margin-right: 20px" /></td>  
<td><a href = "https://github.com/krzjoa/bace"><b>bace</b></a><br>Python implementations of Naive Bayes algorithm variants</td>
</tr>

<tr>
<td><a href="https://github.com/krzjoa/kaggle-metrics"><img src='https://raw.githubusercontent.com/krzjoa/kaggle-metrics/master/img/kmlogo.png' width="100" style="margin-right: 20px" /></td>  
<td><a href = "https://github.com/krzjoa/kaggle-metrics"><b>kaggle-metrics</b></a><br>Metrics for Kaggle competitions</td>
</tr>

<tr>
<td><a href="https://github.com/krzjoa/salto"><img src='https://raw.githubusercontent.com/krzjoa/salto/main/img/salto-logo.png' width="100" style="margin-right: 20px" /></td>  
<td><a href = "https://github.com/krzjoa/salto"><b>salto</b></a><br>Playing with embedding vectors</td>
</tr>


<tr>
<td></td>  
<td><a href = "https://github.com/krzjoa/sciquence"><b>sciquence</b></a> <br>Miscellaneous algorithms for processing sequences and time series</td>
</tr>

<tr>
<td><a href="https://github.com/krzjoa/wolne_lektury"><img src='https://raw.githubusercontent.com/krzjoa/wolne_lektury/main/img/wl_logo.png' width="100" style="margin-right: 20px" /></td>  
<td><a href = "https://github.com/krzjoa/wolne_lektury"><b>wolne_lektury</b></a><br>An unofficial REST API client for Wolne Lektury</td>
</tr>

</thead>
</table>

</div>

<div id="Datasets" class="tabcontent">
<table>
<thead>

<tr>
<td>
<a href = "https://github.com/krzjoa/Komentarze"><b>Komentarze</b></a>
<br>A NLP dataset of Internet comments (in Polish) to filter the hateful ones.
<br>Gathered for my master's thesis project in 2015/2016.

</td>
</tr>

</thead>
</table>
</div>

<script>
document.getElementById("defaultOpen").click();
</script>




