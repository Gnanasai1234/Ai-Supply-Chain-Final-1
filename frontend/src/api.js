import axios from 'axios'

const api = axios.create({ baseURL: `${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/api` })
api.interceptors.request.use(config => {
  const token = sessionStorage.getItem('chainwise_access_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})
api.interceptors.response.use(response => response, error => {
  const path = error.config?.url || ''
  if (error.response?.status === 401 && sessionStorage.getItem('chainwise_access_token') && !path.endsWith('/auth/login')) {
    sessionStorage.removeItem('chainwise_access_token')
    window.location.assign('/')
  }
  return Promise.reject(error)
})
export default api
