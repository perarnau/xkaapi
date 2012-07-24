
#ifndef _KAAPI_VERSION_H_
#define _KAAPI_VERSION_H_

struct kaapi_version_t;

/** Link replicats of the same version
*/
typedef struct kaapi_link_version_t {
  struct kaapi_version_t*       version;        /* the version */
  struct kaapi_link_version_t*  next;           /* the next replicat version */
  kaapi_frame_tasklist_t*       tl;             /* the container of task that acceed the version */
} kaapi_link_version_t;


/** Serves to detect: W -> R dependency or R -> W dependency but not yet cw...
*/
typedef struct kaapi_version_t {
  kaapi_access_mode_t      last_mode;       /* */
  kaapi_data_t*            handle;          /* */
  kaapi_taskdescr_t*       writer_task;     /* last writer task of the version, 0 if no indentify task (input data) */
} kaapi_version_t;


/** Find the version object associated to the addr.
    If not find, then insert into the table a new with 
      last_mode == KAAPI_ACCESS_MODE_VOID.
    The returned object should be initialized correctly 
    with a first initial access not KAAPI_ACCESS_MODE_VOID.
    See kaapi_version_add_initialaccess.
    On return islocal is set to 1 iff the access is local to the thread 'thread'.
    [Not: this is for partitioning into multiple threads; currently not used ]
*/
extern kaapi_version_t* kaapi_version_findinsert( 
    int* islocal,
    struct kaapi_thread_context_t* thread,
    kaapi_frame_tasklist_t*        ftl,
    const void*                    addr 
);

/** Set the initial access of a version.
    The new version is associated with an initial task which
    dependent on the initial access mode (m=R|RM -> task move,
    m=W|CW -> task alloc).
    In case of move, the source data (host pointer) is supposed
    to be recopied into the remote address space.
*/
extern int kaapi_version_add_initialaccess( 
    kaapi_version_t*           ver, 
    kaapi_frame_tasklist_t*    ftl,
    kaapi_access_mode_t        m,
    void*                      data, 
    const kaapi_memory_view_t* view 
);


/**
*/
extern kaapi_version_t* kaapi_version_createreplicat( 
    kaapi_frame_tasklist_t* ftl,
    kaapi_version_t*        master_version
);


/** Insert a synchronization points between the version and the master version
*/
extern int kaapi_thread_insert_synchro( 
    kaapi_frame_tasklist_t* ftl,
    kaapi_version_t*        version, 
    kaapi_access_mode_t     m
);


/** Invalidate all replicats, except version
*/
extern int kaapi_version_invalidatereplicat( 
    kaapi_version_t*     version
);

#endif /* _KAAPI_VERSION_H_ */
