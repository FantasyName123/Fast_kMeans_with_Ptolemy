# Fast kMeans with Ptolemy

Die grundlegende Idee dieses Projektes ist es, mit Hilfe der ptolomäischen Ungleichung
einen schnellen kMeans Algorithmus (oder mehrere) zu entwickeln. Um genauer zu sein,
einen Algorithmus zu entwickeln, der die gleiche Ausgabe wie der Lloyd-Algorithmus
liefert, sowie wie bspw. der Elkan- oder der Hamerly-Algorithmus dies tun.

Grundsätzlich kann oft dort, wo die Dreiecksungleichung als Schranke eingesetzt wird,
auch die ptolomäische Ungleichung eingesetzt werden. Es gibt dabei zwei wesentliche
Punkte zu beachten:
1) Die Metrik muss ptolomäisch sein
2) Wir benötigen ein zusätzliches Pivotelement

Da wir beim kMeans Problem ausschließlich mit der euklidischen Distanz hantieren und
diese ptolomäisch ist, ist 1. gegeben. Der 2. Punkt ist in diesem Kontext wesentlich
interessanter. Darauf möchte ich an dieser Stelle aber nicht eingehen, da dies den
Rahmen dieser Einleitung sprengen würde und hier mehr Fokus auf den Code gelegt werden
soll.

### Disclaimer
Ich habe einige Programmiererfahrungen, ein Profi bin ich jedoch nicht. So kann es sein,
dass mich nicht immer an Best-Practices gehalten und es an einigen Stellen eleganteren
oder effizienteren Code gibt.

Der Code einsatzfähig und einige Untersuchungen können mit seiner Hilfe durchgeführt werden. Er ist
aber keineswegs fertig, es handelt sich viel mehr um ein "Work in Progress" Projekt, das an vielen
Stellen ausgebessert und erweitert werden kann. Ich habe mich bemüht, es in einem möglichst 
übersichtliche Zustand zu übergeben, sodass ich hoffe, es ist möglich, sich nach einiger
Einarbeitungszeit in dem Code zurecht zu finden.

### Der Code
In der Datei "kMeansVariants.py" sind die Algorithmen implementiert. Dort zu finden sind der
Lloyd-Algorithmus, der Elkan-Algorithmus nach Vorlage aus dem Paper "Using the Triangle
Inequality to Accelerate k-Means" und eine modifizierte Version des Elkan-Algorithmus
(elkan_full_ptolemy), bei der die oberen und unteren Schranken mit der ptolemäischen
Ungleichung anstelle der Dreiecksungleichung erstellt werden. Zusätzlich gibt es Methoden,
die die Zahl der im Algorithmus verwendeten Distanzberechnungen zählen. Dies erlaubt es,
die Laufzeit der Methoden zu vergleichen, ohne dass dabei die Zählung der Distanzberechnung
zu Buche schlägt. Da die Zahl der Distanzberechnungen beim Lloyd-Algorithmus sehr einfach
ist, gibt es dafür keine extra Methode. In der Datei "kMeansVariants_old.py" befinden sich
ältere und deutlich langsamere (da weniger vektorisiert) Implementierungen der Algorithmen.
Diese Datei kann noch als Archiv genutzt werden, dort finden sich bereits verschiedene
Ansätze, die ptolomäische Ungleichung in den Elkan-Algorithmus und auch den Hamerly-Algorithmus
zu integrieren.

Ausgeführt werden soll natürlich die "main.py" Datei. Dort wird zunächst einen Datensatz
erstellt oder abgerufen. Anschließend wird ein Satz an initialen Zentren erstellt. Schließlich
werden die Algorithmen aufgerufen, während die benötigte Zeit gestoppt und in der Konsole
ausgegeben wird. Außerdem wird getestet, ob die Ausgabe aller Algorithmen mit der des
Lloyd-Algorithmus übereinstimmen. Die Variable "show_visualisation" kann auf "True" gesetzt
werden, um die Zentren und die Cluster zu plotten. (Das Matplotlib-Fenster muss geschlossen
werden, um im Code fortzufahren.) Die Variable "compare_with_sklearn" kann auf "True" gesetzt
werden, um den Implementierung des Elkan-Algorithmus von sklearn laufen zu lassen und die
benötigte Laufzeit ausgeben zu lassen. Dies soll dazu dienen, zu schauen, ob die Größenordnung
der Laufzeit der selbst implementierten Algorithmen vergleichbar mit der einer Implementierung
aus einer bestehenden Bibliothek ist. Soweit ich es bisher einsehen konnte, ist dies der Fall.
Der Rest, der auskommentiert ist, besteht aus älteren Codezeilen, die voraussichtlich nicht mehr
gebraucht werden. Sicherheitshalber habe ich diese bisher aber nicht gelöscht. Hauptsächlich
bezieht sich der Code auf die älteren Implementierungen der kMeans-Algorithmen. So zeigt der
erste Teil, wie diese aufgerufen werden können.

In der Datei "datasets.py" stehen verschiedene Methoden zur Verfügung, um Datensätze zu generieren.
Bei den Datensätzen, die zufällig erstellt werden, sollte darauf geachtet werden, ob ein fester
Seed verwendet wird, um Reproduzierbarkeit zu gewährleisten. Dies kann sinnvoll sein, um
Reproduzierbarkeit zu gewährleisten. Andererseits kann es auch gewünscht sein, bei jedem Durchlauf
verschiedene Datensätze zu erzeugen, um die statistische Aussagekraft zu erhöhen. Es ist sicherlich
möglich und sinnvoll, weitere Datensätze hinzuzufügen. Die "Birch"-Datensätze wurden einer
Website entnommen. Weitere Informationen dazu in den Docstrings.

Die Dateien "subroutines.py" und "helpers.py" enthalten Subroutinen der kMeans Algorithmen und
nützliche Helferfunktionen, wobei die Methode "all_dist" nur in den älteren kMeans Methoden Anwendung
findet, und daher keine hohe Relevanz hat. Die Methoden in der Datei "scrips.py" sind als Skripte
ausgelegt, um die main Datei aufzuräumen. Sie beziehen sich allerdings ebenfalls noch auf die älteren
Methoden und müssten noch angepasst werden.

### Mögliche Erweiterungen und Fortführungen
Unangenehmerweise gibt es eine Sache, die noch so funktioniert, wie ich mir das vorgestellt habe.
Und zwar sollte der Ablauf der Algorithmen und die Auswahl der Zentren bei der Initialisierung
deterministisch sein, um Reproduzierbarkeit zu gewährleisten. Allerdings kommen bei 
aufeinanderfolgenden Durchläufen verschiedene Anzahlen von Distanzberechnung heraus, was
dem widerspricht. Ich denke, es ist möglich, dieses Problem leicht zu beheben.

Wie bereits angesprochen können weitere Datensätze hinzugefügt werden.

Es gibt eine Vielzahl an Möglichkeiten, wie die ptolomäische Ungleichung in einen kMeans Algorithmus
eingesetzt werden kann. Alle diese Möglichkeiten, auf die ich im nächsten Abschnitt kurz eingehen
werde, könnten in diesem Projekt umgesetzt werden.

Zuletzt können natürlich "Quality of Life Changes" durchgeführt werden und die Ergebnisse sollten ab
irgendeinem Punkt gespeichert werden.

### Einsatzmöglichkeiten der Ptolomäischen Ungleichung
Ich möchte hier als Erstes an meine Ausführungen in dem von mir erstellten, physischen Ordner
verweisen. Dieser Abschnitt ist eher als eine Zusammenfassung dieser Ausführungen anzusehen.

Betrachtet man die Nutzung der Dreiecksungleichung Elkan-Algorithmus zur Erstellung von oberen und
unteren Schranken zwischen einem Datenobjekt und den neuen Zentren, so stellt man fest, dass die dazu
korrespondierenden alten Zentren auf die natürliche Art die Rolle des Pivotelements einnehmen. Für
die Nutzung der ptolomäischen Ungleichung benötigen wir nun ein zweites Pivotelement. Dabei scheint
es (meiner Einschätzung nach), keine solch natürliche Wahl zu geben. Dies zwingt uns selbst eine
Wahl zu treffen, eröffnet damit aber auch ganz neue Möglichkeiten. Darüber hinaus schafft die
Existenz von zwei Ungleichungen gewisse Kombinationsmöglichkeiten. Schließlich ist zu beachten, dass
alleine mit der Dreiecksungleichung schon etliche Verfahren entwickelt wurden, die mit der
ptolomäischen modifiziert werden könnten.

Hier nun eine Aufzählung von möglichen Entscheidungsfragen, deren Antworten nahezu beliebig
miteinander kombiniert werden können:

1) Wie viele (zweite) Pivotelemente sollten genutzt werden?
2) Wie werden diese gewählt?
- Es könnte sich einerseits um fixe Punkte handeln, die für alle Datenpunkte gleich sind. Wie
sollten diese in diesem Fall gewählt werden? Etwa möglichst weit voneinander entfernt? Andererseits
kommt in Frage, dass die Wahl der zweiten Pivotelemente ebenfalls von dem Datenpunkt selbst abhängen.
3) Welche Schranken werden durch die Dreiecksungleichung, welche durch die ptolomäische Ungleichung
und welche durch eine Kombination berechnet?
- Es wäre beispielsweise denkbar, die obere Schranke, von der es nur eine gibt durch die Kombination
beider Ungleichungen zu bestimmen und für die unteren Schranken, von denen es mehr gibt, weiterhin
die Dreiecksungleichung zu benutzen, da die Berechnung beider Ungleichungen teurer ist.
4) Zusätzlich gäbe es evtl. die Möglichkeit, die ptolomäische Ungleichung erst zu berechnen, falls
mit der Dreiecksungleichung kein Ausschluss der Distanzberechnung erzielt werden konnte. (Dies könnte
in der praktischen Umsetzung zB durch Vektorisierung schwierig sein.)
5) Welchen bereits existierenden Algorithmus modifizieren wir mit der ptolomäischen Ungleichung? Gibt
es vielleicht die Möglichkeit, ein gänzlich neuen Algorithmus zu entwerfen?
6) Wie kann der Algorithmus am besten implementiert werden? Gibt es eventuell Optimierungstricks?


