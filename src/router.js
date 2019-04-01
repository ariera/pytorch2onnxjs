import Vue from 'vue';
import Router from 'vue-router';
import Add from './components/add.vue';

Vue.use(Router);

export default new Router({
  routes: [
    {
      path: '/',
      name: 'add',
      component: Add,
    },
    {
      path: '/add',
      name: 'add',
      component: Add,
    },
    {
      path: '/about',
      name: 'about',
      // route level code-splitting
      // this generates a separate chunk (about.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import(/* webpackChunkName: "about" */ './views/About.vue'),
    },
  ],
});
