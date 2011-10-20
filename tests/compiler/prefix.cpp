/// Author: TG  ;o)

#pragma kaapi task read( [__first:__last] ) readwrite( __result[ __last-__first] )
template<typename _InputIterator, typename _OutputIterator>
_OutputIterator partial_sum(_InputIterator __first, _InputIterator __last, _OutputIterator __result)
{
  typedef typename iterator_traits<_InputIterator>::value_type _ValueType;

  if (__first == __last)
    return __result;

  _ValueType __value = *__first++;
  *__result = __value;

  /* Description d'une boucle parallèle:
     - forme canonique décrit par OpenMP, en gros [begin,end, incr]
     - l'interval est toujours découpé de la gauche vers la droite (indice croissant).
     - on répond au voleur toujours de la gauche vers la droite (indice croissant).
     Cet ordre est lié à l'ordre d'itération. Le voleur commencera toujours à voler la
     du côté de la borne indiquée dans le test de la boucle: borne sup si <, <= et incrément; 
     borne inf si > ou >= et décrément.
     
     En fin de boucle, il existe une synchronisation implicite de fin des calculs.
     On peut compléter la fin de la boucle par une clause end parallel loop qui
     donne des actions à faire.
     
     Par défaut, si pas de stealpoint dans la boucle parallèle, alors chaque range
     volé exécute le même pattern du corps de boucle. avec en paramètre de boucle
     le range extrait. L'incrément reste le même.
     
     S'il y a un stealpoint dans le corps de boucle, l'état du flot de control accessible
     dans la clause "stealpoint" correspond à l'état d'une itération i et au point de 
     définition de la clause. Ici avec update de __value ou __result.
  */
  #pragma kaapi parallel loop adaptive("block") 
  for ( ; __first < _last; ++_first )
  {
    /* Sur une requete de vol du travail décrit par KAAPI_RANGE: on appelle la fonction 
       prefix. Celle-çi étant une tâche, cela crée une tâche.
       Le pragma ne concerne donc que le traitement à faire si il y a vol. Le travail
       est décrit par une variable opaque KAAPI_RANGE dont les champs first et last sont
       du même type que la variable d'itération.

       Le code suivant le stealpoint est, dans la grammaire C/C++, une "statement" sans le ';' de la fin.
       Ici, il s'agit d'une "scope statement", i.e. un basic block avec déclaration de variable
       propres au vol et un appel de fonction. La fonction ici correspond à la création d'une tâche puisque
       la fonction est identifiée comme fonction tâche.
       Le scope de déclaration des variables est lié au vol. Ces variables sont accessibles dans la clause merge.
    */
    #pragma kaapi stealpoint { _OutputIterator first_prefix = __result;\
                               _OutputIterator theft_out = partial_sum(KAAPI_INPUT_RANGE.first, KAAPI_INPUT_RANGE.last, __result); }
    __value = __value + *__first;
    *++__result = __value;
  }
  
  /* Ici le merge de la boucle.
     - la clause merge permet de donner du code qui sera exécuter pour chacque requête de vols
     - l'ordre d'exécution est le même que pour la réponse aux requêtes de vol: de la gauche vers 
     la droite en suivant les indices croissants.
     
     Dans cette clause, l'utilisateur peut référencer chaque variable déclarée dans la clause stealpoint.
     Ainsi que la valeur KAAPI_INPUT_RANGE qui, ici, est la copie au moment du vol. Si cette variable
     est modifiée (ou l'un de ses champs) dans la clause stealpoint ou dans le code exécuté, alors ici on
     obtiendra la valeur modifiée (... à voir).
  */
  #pragma kaapi end parallel loop merge { \
      /* calcul du prefix en fin d'interval volé */\
      first_prefix[(KAAPI_INPUT_RANGE.last - KAAPI_INPUT_RANGE.first-1] = \
            first_prefix[(KAAPI_INPUT_RANGE.last - KAAPI_INPUT_RANGE.first-1] + *(first_prefix-1);\
      /* mis à jour du tableau déjà calculé:  x -> x + *(first_prefix-1) qui correspond
         au prefix calculé par le voleur précédent où le code séquentiel.
         Ce code est actuellement séquentiel. Pour le rendre parallèle: déclarer la fonction std::for_each
         comme une fonction tâche, dont l'appel provoquera la création d'une tâche.
       */\
      std::for_each( first_prefix, KAAPI_INPUT_RANGE.last-KAAPI_INPUT_RANGE.first-1, \
            std::bind1st( std::plus, *(first_prefix-1) ) \
      );\
      /* mis à jour du output _result */
      __result = first_prefix + (KAAPI_INPUT_RANGE.last - KAAPI_INPUT_RANGE.first); \
    }
  return ++__result;
}

