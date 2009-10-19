/*
** kaapi_datastructure.h
** ckaapi
** 
** Created on Tue Mar 31 15:19:27 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
** thierry.gautier@imag.fr
** 
** This software is a computer program whose purpose is to execute
** multithreaded computation with data flow synchronization between
** threads.
** 
** This software is governed by the CeCILL-C license under French law
** and abiding by the rules of distribution of free software.  You can
** use, modify and/ or redistribute the software under the terms of
** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
** following URL "http://www.cecill.info".
** 
** As a counterpart to the access to the source code and rights to
** copy, modify and redistribute granted by the license, users are
** provided only with a limited warranty and the software's author,
** the holder of the economic rights, and the successive licensors
** have only limited liability.
** 
** In this respect, the user's attention is drawn to the risks
** associated with loading, using, modifying and/or developing or
** reproducing the software by the user in light of its specific
** status of free software, that may mean that it is complicated to
** manipulate, and that also therefore means that it is reserved for
** developers and experienced professionals having in-depth computer
** knowledge. Users are therefore encouraged to load and test the
** software's suitability as regards their requirements in conditions
** enabling the security of their systems and/or data to be ensured
** and, more generally, to use and operate it in the same conditions
** as regards security.
** 
** The fact that you are presently reading this means that you have
** had knowledge of the CeCILL-C license and that you accept its
** terms.
** 
*/
#ifndef _KAAPI_DATASTRUCTURE_H
#define _KAAPI_DATASTRUCTURE_H 1


/* ========================================================================== */
/** Generic fifo double linked queue
 Declare a fifo queue necessary data structure for a queue structure
 The type should contains the link field named _next
 \param name the name of the typedef
 \param type the type of the queue.
 */
#define KAAPI_QUEUE_FIELD( type )\
type* _next;\
type* _prev

/** Generic queue
 Declare a fifo queue necessary data structure for a queue structure
 The type should contains the link field named _next
 \param name the name of the typedef
 \param type the type of the queue.
 */
#define KAAPI_QUEUE_DECLARE_FIELD( type )\
type* _front;\
type* _back

/** Generic queue
 Declare a fifo queue necessary data structure for a queue structure
 The type should contains the link field named _next
 \param name the name of the typedef
 \param type the type of the queue.
 */
#define KAAPI_QUEUE_DECLARE_FIELD_VOLATILE( type )\
type* volatile _front;\
type* volatile _back

/** Generic fifo queue
 Declare a fifo queue, a double linked list of typed element.
 The type should contains the link field named _next
 \param name the name of the typedef
 \param type the type of the queue.
 */
#define KAAPI_QUEUE_DECLARE( name, type )\
typedef struct name {\
type* _front;\
type* _back;\
} name

/** q: queue*
 */
#define KAAPI_QUEUE_CLEAR( q) (q)->_front = (q)->_back = 0

/** q: queue*
 */
#define KAAPI_QUEUE_EMPTY( q) ((q)->_front == 0)

/** q: queue*
 v: the item to assign
 */
#define KAAPI_QUEUE_FRONT( q ) (q)->_front

/** q: queue*
 v: the item to assign
 */
#define KAAPI_QUEUE_BACK( q ) ((q)->_back)

/** q: queue*
 v: the item to push
 */
#define KAAPI_QUEUE_PUSH_BACK( q, v ) { \
(v)->_next = 0; \
(v)->_prev = (q)->_back; \
if ((q)->_back !=0) (q)->_back->_next = (v);\
else (q)->_front = (v); \
(q)->_back = (v); \
}

/** q: queue*
 v: the item to push
 */
#define KAAPI_QUEUE_PUSH_FRONT( q, v ) { \
(v)->_next = (q)->_front; \
(v)->_prev = 0; \
if ((q)->_front ==0) (q)->_back = (v);\
else (q)->_front->_prev = (v); \
(q)->_front = (v); \
}

/** q: queue*
 v: the result
 */
#define KAAPI_QUEUE_POP_FRONT( q, v ) {\
(v) = (q)->_front; \
(q)->_front = (v)->_next; \
if ((q)->_front ==0) (q)->_back = 0; \
else (q)->_front->_prev = 0; \
}

/** q: queue*
 v: the result
 */
#define KAAPI_QUEUE_POP_BACK( q, v ) {\
(v) = (q)->_back; \
(q)->_back = (v)->_prev; \
if ((q)->_back ==0) (q)->_front = 0; \
else (q)->_back->_next = 0; \
}

/** q: queue*
 v: the item to remove
 */
#define KAAPI_QUEUE_REMOVE( q, v ) {\
if ((v)->_prev !=0) (v)->_prev->_next = (v)->_next;\
else (q)->_front = (v)->_next;\
if ((v)->_next !=0) (v)->_next->_prev = (v)->_prev;\
else (q)->_back = (v)->_prev;\
}

/** Merge l into q
 l should not be empty and becomes empty
 */
#define KAAPI_QUEUE_MERGE_FRONT( q, l) { \
if ((q)->_front ==0) { \
(q)->_front = (l)->_front; \
(q)->_back = (l)->_back;\
}\
else { \
(q)->_front->_prev = (l)->_back; \
(l)->_back->_next = (q)->_front;\
(q)->_front = (l)->_front;\
}\
(l)->_front = (l)->_back = 0;\
}


/* ========================================================================== */
/** Generic fifo queue
 Declare a fifo queue necessary data structure for a queue structure
 The type should contains the link field named _next
 \param name the name of the typedef
 \param type the type of the queue.
 */
#define KAAPI_FIFO_FIELD( type )\
type* _next

/** Generic fifo queue
 Declare a fifo queue necessary data structure for a queue structure
 The type should contains the link field named _next
 \param name the name of the typedef
 \param type the type of the queue.
 */
#define KAAPI_FIFO_DECLARE_FIELD( type )\
type* _front;\
type* _back

/** Generic fifo queue
 Declare a fifo queue, a double linked list of typed element.
 The type should contains the link field named _next
 \param name the name of the typedef
 \param type the type of the queue.
 */
#define KAAPI_FIFO_DECLARE( name, type )\
typedef struct name {\
type* _front;\
type* _back;\
} name;

/** Generic fifo queue
 Declare a fifo queue, a double linked list of typed element.
 The type should contains the link field named _next
 \param name the name of the typedef
 \param type the type of the queue.
 */
#define KAAPI_FIFO_DECLARE_VOLATILE( name, type )\
typedef struct name {\
type* volatile _front;\
type* volatile _back;\
} name;

/** q: queue*
 */
#define KAAPI_FIFO_CLEAR( q) (q)->_front = (q)->_back = 0

/** q: queue*
 */
#define KAAPI_FIFO_EMPTY( q) ((q)->_front == 0)

/** q: queue*
 v: the item to assign
 */
#define KAAPI_FIFO_TOP( q, v ) (v) = (q)->_front

/** q: queue*
 v: the item to assign
 */
#define KAAPI_FIFO_BACK( q ) ((q)->_front)

/** q: queue*
 v: the item to pop
 */
#define KAAPI_FIFO_PUSH( q, v ) { \
(v)->_next = 0; \
if ((q)->_back !=0) (q)->_back->_next = (v);\
else (q)->_front = (v); \
(q)->_back = (v); \
}

/** q: queue*
 v: the result
 */
#define KAAPI_FIFO_POP( q, v ) {\
(v) = (q)->_front; \
(q)->_front = (v)->_next; \
if ((q)->_front ==0) (q)->_back = 0; \
}

/** q: queue*
 p: previous item, pop the next one if p != front(q)
 v: the poped item
 */
#define KAAPI_FIFO_REMOVE( q, p, v ) {\
if ((p) == (q)->_front) (q)->_front = (v)->_next; \
else (p)->_next = (v)->_next; \
if ((q)->_front ==0) (q)->_back = 0; \
}


/* ========================================================================== */
/** Generic stack (lifo order)
 Declare a fifo queue necessary data structure for a stack structure
 The type should contains the link field named _next and _prev
 \param name the name of the typedef
 \param type the type of the queue.
 */
#define KAAPI_STACK_FIELD( type )\
type* _next;\
type* _prev;

/** Generic stack (lifo order)
 Declare a fifo queue necessary data structure for a stack structure
 The type should contains the link field named _next
 \param name the name of the typedef
 \param type the type of the queue.
 */
#define KAAPI_STACK_DECLARE_FIELD( type )\
type* _front; \
type* _back;


/** Generic stack (lifo order)
 Declare a stack, a simple linked list of typed element.
 The type should contains the link field named _next
 \param name the name of the typedef
 \param type the type of the queue.
 */
#define KAAPI_STACK_DECLARE( name, type )\
typedef struct name {\
KAAPI_STACK_DECLARE_FIELD(type);\
} name;

/** q: stack*
 */
#define KAAPI_STACK_CLEAR( q) (q)->_front = (q)->_back = 0;

/** q: stack*
 */
#define KAAPI_STACK_CLEAR_FIELD( q) (q)->_next = (q)->_prev = 0;


/** q: stack*
 */
#define KAAPI_STACK_EMPTY( q) ((q)->_front == 0)

/** q: stack*
 */
#define KAAPI_STACK_BACK( q ) ((q)->_back)

/** q: stack*
 */
#define KAAPI_STACK_TOP( q ) ((q)->_front)

/** q: stack*
 v: the item to push
 */
#define KAAPI_STACK_PUSH( q, v ) \
{ \
(v)->_prev = (q)->_front; \
(v)->_next = 0;\
if ((q)->_front ==0) (q)->_back = (v); \
else (q)->_front->_next = (v);\
(q)->_front = (v);\
}

/** q: stack*
 */
#define KAAPI_STACK_POP( q ) \
{\
(q)->_front = (q)->_front->_prev;\
if ((q)->_front ==0) (q)->_back = 0;\
else (q)->_front->_next =0;\
}

/** q: stack*
 p: previous item, pop the next one if p != front(q)
 v: the poped item
 */
#define KAAPI_STACK_REMOVE( q, p, v ) {\
if ((p) == (q)->_front) (q)->_front = (v)->_prev; \
else (p)->_prev = (v)->_next; \
}

/** Merge l into q
 l should not be empty and becomes empty
 */
#define KAAPI_STACK_MERGE( q, l) { \
if ((q)->_front ==0) { \
(q)->_front = (l)->_front; \
(q)->_back = (l)->_back;\
}\
else { \
(l)->_back->_prev = (q)->_front;\
(q)->_front = (l)->_front;\
}\
(l)->_front = (l)->_back = 0;\
}




#endif // _KAAPI_DATASTRUCTURE_H
