ACKNOWLEDGMENTS

 

First of all I want to thanks everybody that helped me to write the Spanish version of this book. All of them are mentioned in the Acknowledgments of the Spanish book[56].

For this English version, I want to give special thanks to my colleagues that helped me with the translation of this book: Joan Capdevila, Mauro Gómez, Josep Ll. Berral, Katy Wallace and Christopher Bonnett. Without their great support I couldn’t have finished this English version before Easter.

Spanish book acknowledgments:

Escribir un libro requiere motivación pero también mucho tiempo, y por ello quiero empezar agradeciendo a mi familia el apoyo y la comprensión que ha mostrado ante el hecho de que un portátil compartiera con nosotros muchos fines de semana y parte de las vacaciones de Navidad desde que Google anunciara que liberaba TensorFlow el pasado noviembre.

A Oriol Vinyals le quiero agradecer muy sinceramente su disponibilidad y entusiasmo por escribir el prólogo de este libro, que ha sido para mí el primer gran reconocimiento al esfuerzo realizado. A Oriol lo conocí hace un par de años después de intercanviar unos cuantos correos electrónicos y en persona el año pasado. Realmente, un crack del tema de quien nuestro país se debería sentir muy orgulloso e intentar seducirlo para que algún día deje Silicon Valley y venga a Barcelona a fundar aquí nuestro propio Silicon Valley mediterráneo.

Como avanzaba en el prefacio del libro, un antiguo alumno licenciado en físicas e ingeniero en informática, además de haber sido uno de los mejores becarios de investigación que he tenido en el BSC, ha jugado un papel muy importante en esta obra. Se trata de Ferran Julià, que junto con Oriol Núñez han fundado una startup, con sede en mi comarca, en la que se preparan para analizar imágenes con redes neuronales convolucionales, entre otras muchísimas cosas que ofrece UNDERTILE. Este hecho ha permitido que Ferran Julià haya hecho a la perfección el rol de editor en este libro, incidiendo en forma y contenidos, de la misma manera que lo hizo mi editor Llorenç Rubió cuando publiqué con la editorial Libros de Cabecera mi ópera prima.

Ahora bien, a Oriol Núñez le agradezco profundamente la idea que compartió conmigo de ampliar las posibilidades de este libro y hacer llegar sus beneficios a muchas más personas de las que yo tenía en mente originalmente a través de su proyecto conjunto con la Fundació El Maresme para la integración social y la mejora de la calidad de vida de las personas con discapacidad intelectual de mi comarca.

Mi más sincero agradecimiento a todos aquellos que han leído parcial o totalmente esta obra antes de ver la luz. En especial, a un importante data scientist como es Aleix Ruiz de Villa, quién me ha reportado interesantes comentarios para incluir en la versión que tienen en sus manos. Pero también a Mauro Gómez, Oriol Núñez, Bernat García y Manuel Carbonell por sus importantes aportaciones con sus lecturas previas.

Han sido muchos expertos en este tema que no conozco personalmente los que también me han ayudado en este libro, permitiéndome que compartiera sus ideas e incluso sus códigos, y por ello menciono en detalle las fuentes en los apartados correspondientes en este libro, más como muestra de agradecimiento que no para que el lector lo tenga que consultar.

Mi mayor agradecimiento a mi universidad, la Universitat Politècnica de Catalunya – UPC Barcelona Tech, que ha sido el entorno de trabajo que me ha permitido realizar mi investigación sobre estos temas y acumular los conocimientos que aquí quiero compartir. Universidad que además me ofrece dar clases en la Facultat d’Informàtica de Barcelona, a unos alumnos brillantes, quienes me animan a escribir obras como esta.

En la misma línea, quiero agradecer al centro de investigación Barcelona Supercomputing Center – Centro Nacional de Computación (BSC) y en especial a su director Mateo Valero, y los directores de Computer Science Jesús Labarta y Eduard Ayguadé, quienes me han permitido y apoyado siempre esta “dèria” que tengo de tener que estar “parant l’orella” a les tecnologías que vendrán.

Especialmente me gustaría mencionar a dos de mis colegas de la UPC, con quien estoy codo a codo iniciando esta rama de investigación más de “analítica”: Rubèn Tous y Joan Capdevila han mostrado fe ciega en mis “dèrias” de exploración de nuevos temas para conseguir que nuestros conocimientos puedan aportar a esta nueva área llamada High-Performance Big-Data Analytics. Hacía tiempo que no disfrutaba tanto haciendo investigación.

Relacionado con ellos, agradecer a otro gran data scientist, Jesús Cerquides, del Artificial Intelligence Research Institute del CSIC , de quien a través de la codirección de una tesis doctoral estoy descubriendo una nueva y apasionante galaxia en el universo del Machine Learning.

Y no puedo olvidarme de quien aprendo muchísimo, estudiantes que su trabajo final de máster trata estos temas: Sana Imtiaz o Andrea Ferri.

Hablando de GPUs, gracias a Nacho Navarro, responsable del BSC/UPC NVIDIA GPU Center of Excellence, por facilitarme desde el primer momento el uso de sus recursos para “entrenar” a mis redes neuronales.

Mi agradecimiento al catedrático de la UPC Ricard Gavaldà, uno de los mejores data scientist con los que cuenta el país, que con mucha paciencia me llevo de la mano en mis inicios en este inhóspito para mi, pero apasionante, mundo del Machine Learning a mediados del 2006 creando junto con Toni Moreno-García, Josep Ll Berra y Nico Poggi el primer team híbrido de Data Scientist con Computer Engineers, ¡momentos inolvidables!

Gracias a esa experiencia nuestro grupo de investigación incorporó el Machine Learning con resultados tan brillantes como las tesis de Josep Ll. Berral, Javier Alonso o Nico Poggi, donde usábamos el Machine Learning en la gestión de recursos de los complejos sistemas de computación actuales. Desde entonces que me he quedado prendado del Machine Learning.

Pero no fue hasta unos años más tarde, en 2014, cuando con la incorporación al grupo de Jordi Nin y posteriormente de Jose A. Cordero, que no hice el paso adelante en “Deep Learning”. Sin su estímulo por este tema hoy este libro no existiría. Gracias, Jordi y José, y espero que vuestros nuevos retos profesionales os reporten grandes éxitos.

Y no quiero olvidar a Màrius Mollà, quién me empujó a publicar mi primer libro Empresas en la nube. Desde entonces no para de insistir para cuándo la siguiente obra… ¡Pues aquí la tienes! Y a Mauro Cavaller, un gran colaborador de Màrius, cuya contribución fue clave en mi opera prima, me ha aportado esta vez una última revisión formal.

Gracias también a Laura Juan por su exquisita revisión del texto antes de que este viera la luz; sin su ayuda esta obra no tendría la calidad que tiene. A Katy Wallace, por poner nombre a esta colección en la que se edita esta obra. Y a Bernat Torres, por haber creado la fantástica página web de este libro.

A Oriol Pedrera, un genio que domina todas las artes plásticas, que me ha acompañado con mucha paciencia en las diversas técnicas que he ido usando para realizar las ilustraciones del libro, que junto con Júlia Torres y Roser Bellido hemos ido concretando una y otra vez hasta encontrar la versión que encuentran en este libro. ¡Ha sido genial! En la parte artística no quisiera olvidarme del gran ebanista Rafa Jiménez quien acepto, sin rechistar, construirme a medida mi mesa de dibujo.

Agradecer al meetup grup d’estudis de machine learning de Barcelona por acoger la presentación oficial del libro, y a Oriol Pujol per aceptar impartir la conferencia que ha acompañado esta presentación del libro en el meetup.

Y también , muchas gracias a las entidades que me han ayudado a hacer difusión de la existencia de esta obra: la Facultad de Informática de Barcelona (FIB), la aceleradora de proyectos tecnológicos ITNIG, el Col·legi Oficial d’Enginyers Informàtics (COEINF), la Associació d’Antics Alumnes de la FIB (FIBAlumni), el portal de tecnología TECNONEWS, el portal iDigital y el Centre d’Excel·lència en Big Data a Barcelona (Big Data CoE de Barcelona).

Para acabar, una mención especial a la “penya cap als 50”, la “colla dels informàtics” que después de 30 años todavía hacemos encuentros que dejan a uno cargadísimo de energía. El caso es que aquel fin de semana de noviembre que a la gente de Google desde Silicon Valley se le ocurrió sacar a la luz TensorFlow, yo lo pase con esta peña. Si yo no me hubiera cargado las pilas con ellos durante ese fin de semana, les aseguro que al día siguiente, cuando me planteé enfrascarme en escribir este libro no habría tenido la energía necesaria. Gracias Xavier, Lourdes, Carme, Joan, Humbert, Maica, Jordi, Eva, Paqui, Jordi, Lluís, Roser, Ricard y Anna. Ahora ya he agotado la energía, ¿cuándo volvemos a quedar?

[contents link]
